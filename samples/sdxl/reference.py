"""A NumPy transcription of the same VAE, used to check the DirectML graph.

Nothing here runs on the GPU and nothing here is fast -- the convolution
materializes the full im2col matrix -- so it is only usable on small images. Its
job is to be obviously correct, so ``check.py`` has something to compare against
on a machine with no PyTorch installed.

Layer for layer this mirrors ``vae.py``; read the two side by side.
"""

import numpy as np

from vae import (
    BLOCK_OUT_CHANNELS, LATENT_CHANNELS, LAYERS_PER_BLOCK, NORM_EPSILON,
    NORM_GROUPS)


def conv2d(x, weight, bias, stride=1, padding=1, end_padding=None):
    n, _, _, _ = x.shape
    out_channels, _, kh, kw = weight.shape
    end = padding if end_padding is None else end_padding

    padded = np.pad(x, ((0, 0), (0, 0), (padding, end), (padding, end)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]

    _, _, height, width = windows.shape[:4]
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * height * width, -1)
    out = columns @ weight.reshape(out_channels, -1).T + bias
    return out.reshape(n, height, width, out_channels).transpose(0, 3, 1, 2)


def group_norm(x, weight, bias, groups=NORM_GROUPS, epsilon=NORM_EPSILON):
    n, c, h, w = x.shape
    grouped = x.reshape(n, groups, c // groups, h * w)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    variance = grouped.var(axis=(2, 3), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + epsilon)).reshape(n, c, h, w)
    return normalized * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)


def silu(x):
    # tanh form of the sigmoid, which does not overflow for large negative x.
    return x * 0.5 * (1.0 + np.tanh(0.5 * x))


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def linear(x, weight, bias):
    return x @ weight.T + bias


def resnet_block(x, params, prefix, epsilon=NORM_EPSILON):
    h = silu(group_norm(x, params[f"{prefix}.norm1.weight"],
                        params[f"{prefix}.norm1.bias"], epsilon=epsilon))
    h = conv2d(h, params[f"{prefix}.conv1.weight"], params[f"{prefix}.conv1.bias"])
    h = silu(group_norm(h, params[f"{prefix}.norm2.weight"],
                        params[f"{prefix}.norm2.bias"], epsilon=epsilon))
    h = conv2d(h, params[f"{prefix}.conv2.weight"], params[f"{prefix}.conv2.bias"])

    if f"{prefix}.conv_shortcut.weight" in params:
        x = conv2d(x, params[f"{prefix}.conv_shortcut.weight"],
                   params[f"{prefix}.conv_shortcut.bias"], padding=0)
    return x + h


def attention_block(x, params, prefix, epsilon=NORM_EPSILON):
    n, c, h, w = x.shape
    normalized = group_norm(x, params[f"{prefix}.group_norm.weight"],
                            params[f"{prefix}.group_norm.bias"], epsilon=epsilon)
    tokens = normalized.reshape(n, c, h * w).transpose(0, 2, 1)

    query = linear(tokens, params[f"{prefix}.to_q.weight"], params[f"{prefix}.to_q.bias"])
    key = linear(tokens, params[f"{prefix}.to_k.weight"], params[f"{prefix}.to_k.bias"])
    value = linear(tokens, params[f"{prefix}.to_v.weight"], params[f"{prefix}.to_v.bias"])

    scores = query @ key.transpose(0, 2, 1) / np.sqrt(c)
    attended = softmax(scores) @ value
    projected = linear(attended, params[f"{prefix}.to_out.0.weight"],
                       params[f"{prefix}.to_out.0.bias"])
    return projected.transpose(0, 2, 1).reshape(n, c, h, w) + x


def upsample_nearest(x, scale=2):
    return np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)


def mid_block(x, params, prefix):
    x = resnet_block(x, params, f"{prefix}.resnets.0")
    x = attention_block(x, params, f"{prefix}.attentions.0")
    return resnet_block(x, params, f"{prefix}.resnets.1")


def decode(latent, params):
    h = conv2d(latent, params["post_quant_conv.weight"],
               params["post_quant_conv.bias"], padding=0)
    h = conv2d(h, params["decoder.conv_in.weight"], params["decoder.conv_in.bias"])
    h = mid_block(h, params, "decoder.mid_block")

    for i in range(len(BLOCK_OUT_CHANNELS)):
        prefix = f"decoder.up_blocks.{i}"
        for j in range(LAYERS_PER_BLOCK + 1):
            h = resnet_block(h, params, f"{prefix}.resnets.{j}")
        if f"{prefix}.upsamplers.0.conv.weight" in params:
            h = conv2d(upsample_nearest(h), params[f"{prefix}.upsamplers.0.conv.weight"],
                       params[f"{prefix}.upsamplers.0.conv.bias"])

    h = group_norm(h, params["decoder.conv_norm_out.weight"],
                   params["decoder.conv_norm_out.bias"])
    return conv2d(silu(h), params["decoder.conv_out.weight"], params["decoder.conv_out.bias"])


def encode(image, params):
    h = conv2d(image, params["encoder.conv_in.weight"], params["encoder.conv_in.bias"])

    for i in range(len(BLOCK_OUT_CHANNELS)):
        prefix = f"encoder.down_blocks.{i}"
        for j in range(LAYERS_PER_BLOCK):
            h = resnet_block(h, params, f"{prefix}.resnets.{j}")
        if f"{prefix}.downsamplers.0.conv.weight" in params:
            h = conv2d(h, params[f"{prefix}.downsamplers.0.conv.weight"],
                       params[f"{prefix}.downsamplers.0.conv.bias"],
                       stride=2, padding=0, end_padding=1)

    h = mid_block(h, params, "encoder.mid_block")
    h = group_norm(h, params["encoder.conv_norm_out.weight"],
                   params["encoder.conv_norm_out.bias"])
    h = conv2d(silu(h), params["encoder.conv_out.weight"], params["encoder.conv_out.bias"])
    moments = conv2d(h, params["quant_conv.weight"], params["quant_conv.bias"], padding=0)
    return moments[:, :LATENT_CHANNELS]
