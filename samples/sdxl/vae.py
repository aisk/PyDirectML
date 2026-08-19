"""The SDXL variational autoencoder, built as DirectML graphs.

SDXL diffuses in a latent space eight times smaller than the image on each side.
The decoder turns a ``[1, 4, H/8, W/8]`` latent into a ``[1, 3, H, W]`` image; the
encoder does the reverse and is what lets this sample check itself without a
reference framework installed.

Architecture and weight names follow ``diffusers.AutoencoderKL`` with the config
published at ``stabilityai/sdxl-vae``.
"""

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
