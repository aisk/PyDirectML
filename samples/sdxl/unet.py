"""SDXL's UNet, built as a DirectML graph.

This is the model the sampler calls at every step, and the only part of SDXL
that runs more than once per image. It takes a noisy latent, a timestep, and the
text conditioning, and predicts the noise in the latent.

Structurally it is the VAE's encoder and decoder joined at the waist, with two
additions. Every resnet gets the timestep embedding added to it, which is how the
same weights behave differently at different noise levels. And at the two lower
resolutions every resnet is followed by a stack of transformer blocks, each of
which attends over the image and then cross-attends to the text -- that is where
the prompt reaches the pixels.

It runs at half precision. At float32 the weights alone are 10.3 GiB.
"""

import math

import numpy as np

import directml as dml

from dml_layers import (
    Model, broadcast, conv2d, crop_to, diffusers_attention, geglu, group_norm,
    layer_norm, linear, silu, to_channels, to_image, to_tokens,
    upsample_nearest)

LATENT_CHANNELS = 4
BLOCK_OUT_CHANNELS = (320, 640, 1280)
LAYERS_PER_BLOCK = 2
# Per resolution, going down. Level 0 has no attention at all.
TRANSFORMER_LAYERS = (0, 2, 10)
ATTENTION_HEADS = (5, 10, 20)
CROSS_ATTENTION_DIM = 2048
TIME_EMBED_DIM = 1280
ADDITION_TIME_EMBED_DIM = 256
NORM_GROUPS = 32
NORM_EPSILON = 1e-5
# Transformer2DModel hardcodes its GroupNorm epsilon, and it is not the one the
# resnets use.
TRANSFORMER_NORM_EPSILON = 1e-6

# DirectML lays a gemm's weight out for the shape it is about to be multiplied
# by, and when the row count is not a multiple of this it keeps a second,
# repacked copy in the graph's persistent resource instead of reusing the one it
# already has. The rows are tokens, so this is a property of the image size.
GEMM_ROW_ALIGNMENT = 64


def deepest_tokens(height, width):
    """Tokens the 1280-wide level sees, which is the row count of its gemms."""
    def extent(size):
        # The latent, halved once per downsampled level. A stride-2 convolution
        # with one pixel of padding rounds up.
        size //= 8
        for _ in range(len(BLOCK_OUT_CHANNELS) - 1):
            size = -(-size // 2)
        return size

    return extent(height) * extent(width)


def weights_are_duplicated(height, width):
    """Whether this size makes DirectML keep a second copy of the widest weights.

    Thirty of the UNet's seventy transformer layers are 1280 wide, and a second
    copy of them costs 1.7 GiB in each of the two graphs -- 5.4 GiB of weights
    becoming 8.8, which on a 16 GiB card is the difference between a size that
    works and one that hangs the device. 1024x1408 needs less memory than
    1024x1360 for this reason, despite being the larger image.
    """
    return deepest_tokens(height, width) % GEMM_ROW_ALIGNMENT != 0


def nearby_aligned(height, width, reach=256):
    """The closest height to ``height`` that does not, or None if there is none."""
    for offset in range(8, reach + 1, 8):
        for candidate in (height - offset, height + offset):
            if candidate > 0 and not weights_are_duplicated(candidate, width):
                return candidate
    return None


def timestep_embedding(timesteps, dim, flip_sin_to_cos=True, freq_shift=0, max_period=10000):
    """Sinusoidal timestep features, computed on the CPU.

    There are no weights here, so there is nothing to gain from putting it in the
    graph -- and the timestep changes every step, which would make it a graph
    input either way.
    """
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(half, dtype=np.float32) / (half - freq_shift)
    angles = np.asarray(timesteps, np.float32).reshape(-1, 1) * np.exp(exponent).reshape(1, -1)

    embedding = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    if flip_sin_to_cos:
        embedding = np.concatenate([embedding[:, half:], embedding[:, :half]], axis=-1)
    return embedding


def conditioning(timestep, pooled_embeds, original_size, crop, target_size):
    """The two vectors the graph wants alongside the latent.

    SDXL conditions on more than the timestep: the resolution it was cropped
    from, the crop offset, and the resolution being targeted all go in as
    sinusoids, concatenated with the pooled text embedding.
    """
    time_input = timestep_embedding([timestep], BLOCK_OUT_CHANNELS[0])

    time_ids = [*original_size, *crop, *target_size]
    time_ids_embedding = timestep_embedding(time_ids, ADDITION_TIME_EMBED_DIM).reshape(1, -1)
    add_input = np.concatenate([pooled_embeds.reshape(1, -1), time_ids_embedding], axis=-1)

    return (time_input.reshape(1, 1, 1, -1).astype(np.float32),
            add_input.reshape(1, 1, 1, -1).astype(np.float32))


def _embedding_mlp(model, x, params, prefix):
    """Linear, SiLU, Linear -- how both embeddings reach the resnets' width."""
    x = linear(model, x, params[f"{prefix}.linear_1.weight"], params[f"{prefix}.linear_1.bias"])
    return linear(model, silu(x), params[f"{prefix}.linear_2.weight"],
                  params[f"{prefix}.linear_2.bias"])


def resnet_block(model, x, temb, params, prefix):
    """The VAE's resnet plus a timestep term added between the two convolutions."""
    h = conv2d(model, silu(group_norm(model, x, params[f"{prefix}.norm1.weight"],
                                      params[f"{prefix}.norm1.bias"],
                                      groups=NORM_GROUPS, epsilon=NORM_EPSILON)),
               params[f"{prefix}.conv1.weight"], params[f"{prefix}.conv1.bias"])

    projected = linear(model, silu(temb), params[f"{prefix}.time_emb_proj.weight"],
                       params[f"{prefix}.time_emb_proj.bias"])
    h = h + broadcast(to_channels(projected), h.shape)

    h = conv2d(model, silu(group_norm(model, h, params[f"{prefix}.norm2.weight"],
                                      params[f"{prefix}.norm2.bias"],
                                      groups=NORM_GROUPS, epsilon=NORM_EPSILON)),
               params[f"{prefix}.conv2.weight"], params[f"{prefix}.conv2.bias"])

    if f"{prefix}.conv_shortcut.weight" in params:
        x = conv2d(model, x, params[f"{prefix}.conv_shortcut.weight"],
                   params[f"{prefix}.conv_shortcut.bias"], padding=0)
    return x + h


def transformer_block(model, x, context, params, prefix, heads):
    """Attend over the image, cross-attend to the text, then a gated feed-forward."""
    x = x + diffusers_attention(
        model, layer_norm(model, x, params[f"{prefix}.norm1.weight"],
                          params[f"{prefix}.norm1.bias"]),
        params, f"{prefix}.attn1", heads)

    x = x + diffusers_attention(
        model, layer_norm(model, x, params[f"{prefix}.norm2.weight"],
                          params[f"{prefix}.norm2.bias"]),
        params, f"{prefix}.attn2", heads, context)

    return x + geglu(model, layer_norm(model, x, params[f"{prefix}.norm3.weight"],
                                       params[f"{prefix}.norm3.bias"]),
                     params, f"{prefix}.ff")


def transformer_2d(model, x, context, params, prefix, heads, layers):
    """A stack of transformer blocks wrapped in a norm, a reshape and a residual."""
    _, _, height, width = x.shape
    residual = x

    tokens = to_tokens(group_norm(model, x, params[f"{prefix}.norm.weight"],
                                  params[f"{prefix}.norm.bias"],
                                  groups=NORM_GROUPS, epsilon=TRANSFORMER_NORM_EPSILON))
    tokens = linear(model, tokens, params[f"{prefix}.proj_in.weight"],
                    params[f"{prefix}.proj_in.bias"])

    for i in range(layers):
        tokens = transformer_block(model, tokens, context, params,
                                   f"{prefix}.transformer_blocks.{i}", heads)

    tokens = linear(model, tokens, params[f"{prefix}.proj_out.weight"],
                    params[f"{prefix}.proj_out.bias"])
    return to_image(tokens, height, width) + residual


def down_block(model, x, temb, context, params, prefix, heads, layers, downsample):
    """Resnets, optionally each followed by transformers, then a stride-2 convolution."""
    skips = []
    for i in range(LAYERS_PER_BLOCK):
        x = resnet_block(model, x, temb, params, f"{prefix}.resnets.{i}")
        if layers:
            x = transformer_2d(model, x, context, params, f"{prefix}.attentions.{i}",
                               heads, layers)
        skips.append(x)

    if downsample:
        x = conv2d(model, x, params[f"{prefix}.downsamplers.0.conv.weight"],
                   params[f"{prefix}.downsamplers.0.conv.bias"], stride=2, padding=1)
        skips.append(x)
    return x, skips


def up_block(model, x, skips, temb, context, params, prefix, heads, layers, upsample):
    """The mirror image, taking one skip connection per resnet."""
    for i in range(LAYERS_PER_BLOCK + 1):
        x = dml.join([x, skips.pop()], axis=1)
        x = resnet_block(model, x, temb, params, f"{prefix}.resnets.{i}")
        if layers:
            x = transformer_2d(model, x, context, params, f"{prefix}.attentions.{i}",
                               heads, layers)

    if upsample:
        # A level that halved an odd extent on the way down cannot get back to it
        # by doubling, so the upsample is trimmed to whatever the next skip
        # connection is, before the convolution rather than after -- which is
        # what diffusers does by handing Upsample2D an output size.
        _, _, height, width = skips[-1].shape
        x = conv2d(model, crop_to(upsample_nearest(x), height, width),
                   params[f"{prefix}.upsamplers.0.conv.weight"],
                   params[f"{prefix}.upsamplers.0.conv.bias"])
    return x


def build_down(model, params, latent_shape, tokens):
    """Everything up to and including the mid block.

    Returns (inputs, outputs). The outputs are what the second half needs: the
    mid-block result, the timestep embedding, and every skip connection.
    """
    batch = latent_shape[0]
    sample = model.placeholder(latent_shape)
    time_input = model.placeholder([batch, 1, 1, BLOCK_OUT_CHANNELS[0]])
    add_input = model.placeholder(
        [batch, 1, 1, TIME_EMBED_DIM + 6 * ADDITION_TIME_EMBED_DIM])
    context = model.placeholder([batch, 1, tokens, CROSS_ATTENTION_DIM])

    temb = (_embedding_mlp(model, time_input, params, "time_embedding")
            + _embedding_mlp(model, add_input, params, "add_embedding"))

    x = conv2d(model, sample, params["conv_in.weight"], params["conv_in.bias"])
    skips = [x]

    levels = len(BLOCK_OUT_CHANNELS)
    for i in range(levels):
        x, produced = down_block(model, x, temb, context, params, f"down_blocks.{i}",
                                 ATTENTION_HEADS[i], TRANSFORMER_LAYERS[i],
                                 downsample=i != levels - 1)
        skips.extend(produced)

    x = resnet_block(model, x, temb, params, "mid_block.resnets.0")
    x = transformer_2d(model, x, context, params, "mid_block.attentions.0",
                       ATTENTION_HEADS[-1], TRANSFORMER_LAYERS[-1])
    x = resnet_block(model, x, temb, params, "mid_block.resnets.1")

    return (sample, time_input, add_input, context), [x, temb] + skips


def build_up(model, params, mid_shape, temb_shape, skip_shapes, tokens):
    """The second half: the up blocks and the output convolution."""
    x = model.placeholder(mid_shape)
    temb = model.placeholder(temb_shape)
    context = model.placeholder([mid_shape[0], 1, tokens, CROSS_ATTENTION_DIM])
    skips = [model.placeholder(shape) for shape in skip_shapes]

    levels = len(BLOCK_OUT_CHANNELS)
    for i in range(levels):
        level = levels - 1 - i
        x = up_block(model, x, skips, temb, context, params, f"up_blocks.{i}",
                     ATTENTION_HEADS[level], TRANSFORMER_LAYERS[level],
                     upsample=i != levels - 1)

    x = group_norm(model, x, params["conv_norm_out.weight"], params["conv_norm_out.bias"],
                   groups=NORM_GROUPS, epsilon=NORM_EPSILON)
    return conv2d(model, silu(x), params["conv_out.weight"], params["conv_out.bias"])


class UNet:
    """The UNet as two compiled graphs, because one will not fit.

    DirectML folds a model's OWNED_BY_DML weights into a single persistent
    buffer, and a single D3D12 buffer stops at 4 GiB -- allocating one past that
    removes the device rather than failing. The UNet is 4.78 GiB at half
    precision. Split at the mid block it is 2.31 and 2.47 GiB, and the tensors
    that cross between the halves -- the mid-block result, the timestep
    embedding, and nine skip connections -- come to about 54 MiB at 1024x1024,
    which is a millisecond of PCIe each way.
    """

    def __init__(self, device, params, height, width,
                 dtype=np.float16, tokens=77, batch=1):
        self.batch = batch
        latent_shape = [batch, LATENT_CHANNELS, height // 8, width // 8]

        self.down = Model(device, dtype)
        _, outputs = build_down(self.down, params, latent_shape, tokens)
        shapes = [list(o.shape) for o in outputs]
        self.down.compile(outputs)

        self.up = Model(device, dtype)
        noise = build_up(self.up, params, shapes[0], shapes[1], shapes[2:], tokens)
        self.up.compile([noise])

    @property
    def input_count(self):
        return self.down.input_count + self.up.input_count

    @property
    def temporary_size(self):
        # One temporary buffer belongs to the device and is resized to whatever
        # the dispatch in front of it asks for, and the halves are dispatched one
        # after the other, so the peak is the larger of the two rather than a sum.
        return max(self.down.temporary_size, self.up.temporary_size)

    @property
    def persistent_size(self):
        return self.down.persistent_size + self.up.persistent_size

    def __call__(self, latent, time_input, add_input, context):
        """Predict the noise in ``latent``."""
        mid, temb, *skips = self.down.run(latent, time_input, add_input, context)
        noise, = self.up.run(mid, temb, context, *skips)
        return noise
