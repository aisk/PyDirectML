"""The SDXL variational autoencoder, built as DirectML graphs.

SDXL diffuses in a latent space eight times smaller than the image on each side.
The decoder turns a ``[1, 4, H/8, W/8]`` latent into a ``[1, 3, H, W]`` image; the
encoder does the reverse and is what lets this sample check itself without a
reference framework installed.

Architecture and weight names follow ``diffusers.AutoencoderKL`` with the config
published at ``stabilityai/sdxl-vae``.
"""

import numpy as np

import directml as dml

from dml_layers import (
    Model, attention_block, conv2d, group_norm, resnet_block, silu, sizes,
    upsample_nearest)

BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
LAYERS_PER_BLOCK = 2
NORM_GROUPS = 32
NORM_EPSILON = 1e-6
LATENT_CHANNELS = 4
SCALE_FACTOR = 8

# Latents come out of the encoder with roughly unit variance only after this
# scaling; the diffusion model is trained in the scaled space.
SCALING_FACTOR = 0.13025


def _mid_block(model, x, params, prefix):
    """Two resnets with self-attention between them, at the lowest resolution."""
    x = resnet_block(model, x, params, f"{prefix}.resnets.0", epsilon=NORM_EPSILON)
    x = attention_block(model, x, params, f"{prefix}.attentions.0", epsilon=NORM_EPSILON)
    return resnet_block(model, x, params, f"{prefix}.resnets.1", epsilon=NORM_EPSILON)


def build_decoder(model, params, latent_shape):
    """Add the decoder to ``model``. Returns (latent input, image output)."""
    latent = model.placeholder(latent_shape)

    h = conv2d(model, latent, params["post_quant_conv.weight"],
               params["post_quant_conv.bias"], padding=0)
    h = conv2d(model, h, params["decoder.conv_in.weight"], params["decoder.conv_in.bias"])
    h = _mid_block(model, h, params, "decoder.mid_block")

    for i in range(len(BLOCK_OUT_CHANNELS)):
        prefix = f"decoder.up_blocks.{i}"
        # One more resnet per block than the encoder has, which is what
        # layers_per_block + 1 means in the diffusers config.
        for j in range(LAYERS_PER_BLOCK + 1):
            h = resnet_block(model, h, params, f"{prefix}.resnets.{j}", epsilon=NORM_EPSILON)
        if f"{prefix}.upsamplers.0.conv.weight" in params:
            h = conv2d(model, upsample_nearest(h),
                       params[f"{prefix}.upsamplers.0.conv.weight"],
                       params[f"{prefix}.upsamplers.0.conv.bias"])

    h = group_norm(model, h, params["decoder.conv_norm_out.weight"],
                   params["decoder.conv_norm_out.bias"],
                   groups=NORM_GROUPS, epsilon=NORM_EPSILON)
    image = conv2d(model, silu(h), params["decoder.conv_out.weight"],
                   params["decoder.conv_out.bias"])
    return latent, image


def build_encoder(model, params, image_shape):
    """Add the encoder to ``model``. Returns (image input, latent mean output).

    The encoder predicts a Gaussian per latent pixel; this keeps the mean and
    drops the log-variance, which is what ``DiagonalGaussianDistribution.mode()``
    does and what image-to-image pipelines use.
    """
    image = model.placeholder(image_shape)

    h = conv2d(model, image, params["encoder.conv_in.weight"], params["encoder.conv_in.bias"])

    for i in range(len(BLOCK_OUT_CHANNELS)):
        prefix = f"encoder.down_blocks.{i}"
        for j in range(LAYERS_PER_BLOCK):
            h = resnet_block(model, h, params, f"{prefix}.resnets.{j}", epsilon=NORM_EPSILON)
        if f"{prefix}.downsamplers.0.conv.weight" in params:
            # Downsample2D pads the bottom and right edges only, then strides by
            # two with no padding of its own.
            h = conv2d(model, h, params[f"{prefix}.downsamplers.0.conv.weight"],
                       params[f"{prefix}.downsamplers.0.conv.bias"],
                       stride=2, padding=0, end_padding=1)

    h = _mid_block(model, h, params, "encoder.mid_block")
    h = group_norm(model, h, params["encoder.conv_norm_out.weight"],
                   params["encoder.conv_norm_out.bias"],
                   groups=NORM_GROUPS, epsilon=NORM_EPSILON)
    h = conv2d(model, silu(h), params["encoder.conv_out.weight"],
               params["encoder.conv_out.bias"])
    moments = conv2d(model, h, params["quant_conv.weight"], params["quant_conv.bias"], padding=0)

    _, _, height, width = sizes(moments)
    mean = dml.slice(moments, [0, 0, 0, 0], [1, LATENT_CHANNELS, height, width], [1, 1, 1, 1])
    return image, mean


def decoder(device, params, height, width):
    """Compile a decoder that produces a ``height`` by ``width`` image."""
    model = Model(device)
    latent_shape = [1, LATENT_CHANNELS, height // SCALE_FACTOR, width // SCALE_FACTOR]
    _, image = build_decoder(model, params, latent_shape)
    return model.compile([image])


def encoder(device, params, height, width):
    """Compile an encoder that consumes a ``height`` by ``width`` image."""
    model = Model(device)
    _, latent = build_encoder(model, params, [1, 3, height, width])
    return model.compile([latent])


# --------------------------------------------------------------------------
# Decoding in tiles

# The decoder is the largest single allocation a generation makes -- 8.45 GiB of
# scratch at 1024x1344, against the UNet's 1.53 -- because every intermediate is
# a full-size image and the widest of them carry 512 channels. Decoding one tile
# at a time replaces that with a fixed cost per tile.
#
# It is an approximation, not a refactoring. GroupNorm normalizes over the whole
# feature map, so a tile decoded alone sees different statistics than the same
# region does inside a whole-image decode. Overlapping the tiles and crossfading
# between them is what hides the difference, and is what diffusers does as well.
TILE_LATENT = 64
TILE_OVERLAP = 8


def _tile_starts(extent, tile, overlap):
    """Evenly spaced tile origins covering ``extent``, every tile the same size.

    Even spacing rather than a fixed stride with a short tile at the end: one
    compiled graph then serves every tile, and the seams come out equally wide
    instead of leaving the last one much narrower than the rest.
    """
    if extent <= tile:
        return [0]
    count = -(-(extent - overlap) // (tile - overlap))
    span = extent - tile
    return [round(i * span / (count - 1)) for i in range(count)]


def _crossfade(starts, tile):
    """Per-axis weights: a linear ramp across whatever the tiles actually share."""
    weight = np.ones(tile, np.float32)
    if len(starts) > 1:
        overlap = tile - min(b - a for a, b in zip(starts, starts[1:]))
        ramp = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        weight[:overlap] = ramp
        weight[tile - overlap:] = ramp[::-1]
    return weight


class TiledDecoder:
    """A decoder that runs one tile-sized graph over the whole latent.

    Same ``run`` as the whole-image decoder: one latent in, one image out. The
    graph is compiled once and dispatched per tile, so what grows with the image
    is the number of dispatches rather than the size of any allocation.
    """

    def __init__(self, device, params, height, width,
                 tile=TILE_LATENT, overlap=TILE_OVERLAP):
        self.height, self.width = height, width
        self.tile = min(tile, height // SCALE_FACTOR, width // SCALE_FACTOR)
        overlap = min(overlap, self.tile // 2)

        self.rows = _tile_starts(height // SCALE_FACTOR, self.tile, overlap)
        self.columns = _tile_starts(width // SCALE_FACTOR, self.tile, overlap)

        side = self.tile * SCALE_FACTOR
        self.model = decoder(device, params, side, side)
        self.window = np.outer(
            _crossfade([row * SCALE_FACTOR for row in self.rows], side),
            _crossfade([column * SCALE_FACTOR for column in self.columns], side))

    @property
    def tiles(self):
        return len(self.rows) * len(self.columns)

    @property
    def input_count(self):
        return self.model.input_count

    @property
    def temporary_size(self):
        return self.model.temporary_size

    @property
    def persistent_size(self):
        return self.model.persistent_size

    def run(self, latent):
        side = self.tile * SCALE_FACTOR
        image = np.zeros((1, 3, self.height, self.width), np.float32)
        weight = np.zeros((1, 1, self.height, self.width), np.float32)

        for top in self.rows:
            for left in self.columns:
                piece, = self.model.run(
                    latent[:, :, top:top + self.tile, left:left + self.tile])
                y, x = top * SCALE_FACTOR, left * SCALE_FACTOR
                image[:, :, y:y + side, x:x + side] += piece * self.window
                weight[:, :, y:y + side, x:x + side] += self.window

        # Two crossfades meeting sum to one, but a ramp on the outer border has
        # no neighbour to complete it, so the weights are divided back out.
        return [image / weight]
