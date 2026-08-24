"""SDXL weights under LDM's key names, read as the names these graphs want.

The checkpoints ComfyUI and A1111 load are a single file holding all three
models under the key names of Stability's latent-diffusion code: the top-level
prefixes ``model.diffusion_model``, ``first_stage_model`` and
``conditioner.embedders`` are the attributes of its ``DiffusionEngine``. These
graphs follow diffusers' names instead, so everything here is renaming --
dictionary in, dictionary out, no file access and no DirectML.

Every function takes tensors with the top-level prefix already stripped, which
is how ``weights.py`` reads them.

Three of the renames are more than a name:

* OpenCLIP keeps one fused QKV matrix per layer where HF keeps three.
* OpenCLIP's text projection is stored the way it is used, ``x @ W``; HF stores
  an ``nn.Linear`` weight, which is its transpose.
* The VAE's attention projections are 1x1 convolutions here and plain linears in
  diffusers, so they lose two trailing axes.

Anything unrecognized raises rather than being dropped. A key this does not know
about means the table is wrong or the checkpoint is not SDXL, and both are worth
hearing about before a graph is built out of half the weights.
"""

import re

import numpy as np

# The VAE has four resolutions, and the decoder's are stored lowest-first.
VAE_LEVELS = 4


def _convert(tensors, rule):
    """Apply ``rule`` to every tensor. A rule returns zero or more (name, value)."""
    converted = {}
    for name, tensor in tensors.items():
        for renamed, value in rule(name, tensor):
            converted[renamed] = value
    return converted


# --------------------------------------------------------------------------
# The UNet

# ResBlock keeps its layers under the Sequentials it built them in.
_RESNET = {
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "emb_layers.1": "time_emb_proj",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "skip_connection": "conv_shortcut",
}

_UNET_TOP = {
    "input_blocks.0.0": "conv_in",
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "label_emb.0.0": "add_embedding.linear_1",
    "label_emb.0.2": "add_embedding.linear_2",
    "out.0": "conv_norm_out",
    "out.2": "conv_out",
}

_UNET_BLOCK = re.compile(r"(input_blocks|output_blocks|middle_block)\.(\d+)\.(?:\d+\.)?(.+)")


def _unet_key(name):
    """The diffusers name for one UNet tensor.

    LDM numbers the down and up paths as one flat list of blocks, so the
    resolution level and the layer within it have to be divided back out. Which
    kind of block a tensor belongs to is decided by its own name rather than by
    its position, because the sub-index that would say so shifts depending on
    whether the level has attention at all.
    """
    path, _, param = name.rpartition(".")

    if path in _UNET_TOP:
        return f"{_UNET_TOP[path]}.{param}"

    match = _UNET_BLOCK.fullmatch(path)
    if not match:
        raise ValueError(f"unrecognized UNet key: {name}")
    where, index, rest = match.group(1), int(match.group(2)), match.group(3)

    if where == "middle_block":
        # Resnet, transformer, resnet.
        if index == 1:
            return f"mid_block.attentions.0.{rest}.{param}"
        return f"mid_block.resnets.{index // 2}.{_RESNET[rest]}.{param}"

    # The down path starts at input_blocks.1, because block 0 is conv_in.
    down = where == "input_blocks"
    block, layer = divmod(index - 1 if down else index, 3)

    if rest == "op":
        return f"down_blocks.{block}.downsamplers.0.conv.{param}"
    if rest == "conv":
        return f"up_blocks.{block}.upsamplers.0.conv.{param}"

    side = "down" if down else "up"
    if rest in _RESNET:
        return f"{side}_blocks.{block}.resnets.{layer}.{_RESNET[rest]}.{param}"
    if rest in ("norm", "proj_in", "proj_out") or rest.startswith("transformer_blocks."):
        return f"{side}_blocks.{block}.attentions.{layer}.{rest}.{param}"
    raise ValueError(f"unrecognized UNet key: {name}")


def unet(tensors):
    """``model.diffusion_model.*`` as ``diffusers.UNet2DConditionModel``."""
    return _convert(tensors, lambda name, tensor: [(_unet_key(name), tensor)])


# --------------------------------------------------------------------------
# The VAE

_VAE_ATTENTION = {"norm": "group_norm", "q": "to_q", "k": "to_k", "v": "to_v",
                  "proj_out": "to_out.0"}

_VAE_BLOCK = re.compile(r"(encoder|decoder)\.(down|up)\.(\d+)\.(.+)")
_VAE_MID = re.compile(r"(encoder|decoder)\.mid\.(block_1|block_2|attn_1)\.(.+)")
_VAE_CONV = re.compile(r"(encoder|decoder)\.conv_(in|out)")
_VAE_NORM = re.compile(r"(encoder|decoder)\.norm_out")


def _vae_key(name):
    """The diffusers name for one VAE tensor."""
    path, _, param = name.rpartition(".")

    match = _VAE_BLOCK.fullmatch(path)
    if match:
        half, direction, index, rest = match.groups()
        index = int(index)
        if direction == "up":
            # diffusers numbers the decoder's blocks in the order it runs them.
            index = VAE_LEVELS - 1 - index
        prefix = f"{half}.{direction}_blocks.{index}"

        if rest in ("downsample.conv", "upsample.conv"):
            return f"{prefix}.{direction}samplers.0.conv.{param}"
        block, layer, inner = rest.split(".", 2)
        if block != "block":
            raise ValueError(f"unrecognized VAE key: {name}")
        inner = "conv_shortcut" if inner == "nin_shortcut" else inner
        return f"{prefix}.resnets.{int(layer)}.{inner}.{param}"

    match = _VAE_MID.fullmatch(path)
    if match:
        half, which, rest = match.groups()
        if which == "attn_1":
            return f"{half}.mid_block.attentions.0.{_VAE_ATTENTION[rest]}.{param}"
        return f"{half}.mid_block.resnets.{int(which[-1]) - 1}.{rest}.{param}"

    if path in ("quant_conv", "post_quant_conv") or _VAE_CONV.fullmatch(path):
        return f"{path}.{param}"
    match = _VAE_NORM.fullmatch(path)
    if match:
        return f"{match.group(1)}.conv_norm_out.{param}"
    raise ValueError(f"unrecognized VAE key: {name}")


def vae(tensors):
    """``first_stage_model.*`` as ``diffusers.AutoencoderKL``."""
    def rule(name, tensor):
        renamed = _vae_key(name)
        if ".attentions." in renamed and tensor.ndim == 4:
            tensor = tensor.reshape(tensor.shape[:2])
        return [(renamed, tensor)]

    return _convert(tensors, rule)


# --------------------------------------------------------------------------
# The two text encoders

_CLIP_G_TOP = {
    "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "positional_embedding": "text_model.embeddings.position_embedding.weight",
    "ln_final.weight": "text_model.final_layer_norm.weight",
    "ln_final.bias": "text_model.final_layer_norm.bias",
}

_CLIP_G_LAYER = {
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
    "attn.out_proj": "self_attn.out_proj",
}

_CLIP_G_BLOCK = re.compile(r"transformer\.resblocks\.(\d+)\.(.+)")


def _clip_g(tensors):
    """OpenCLIP ViT-bigG as ``transformers.CLIPTextModelWithProjection``."""
    def rule(name, tensor):
        if name in _CLIP_G_TOP:
            return [(_CLIP_G_TOP[name], tensor)]
        if name == "text_projection":
            # Stored the way it is used, ``x @ W``; nn.Linear holds the transpose.
            return [("text_projection.weight", np.ascontiguousarray(tensor.T))]
        if name == "logit_scale":
            return []  # Only the image tower it was trained against uses this.

        match = _CLIP_G_BLOCK.fullmatch(name)
        if not match:
            raise ValueError(f"unrecognized OpenCLIP key: {name}")
        layer, rest = f"text_model.encoder.layers.{match.group(1)}", match.group(2)

        if rest in ("attn.in_proj_weight", "attn.in_proj_bias"):
            # One fused matrix where HF keeps three, stacked query, key, value.
            param, width = rest.rsplit("_", 1)[1], tensor.shape[0] // 3
            return [(f"{layer}.self_attn.{projection}_proj.{param}",
                     np.ascontiguousarray(tensor[i * width:(i + 1) * width]))
                    for i, projection in enumerate("qkv")]

        path, _, param = rest.rpartition(".")
        if path in _CLIP_G_LAYER:
            return [(f"{layer}.{_CLIP_G_LAYER[path]}.{param}", tensor)]
        raise ValueError(f"unrecognized OpenCLIP key: {name}")

    return _convert(tensors, rule)


def _clip_l(tensors):
    """CLIP ViT-L, which the checkpoint already stores under HF's own names."""
    return {name: tensor for name, tensor in tensors.items()
            if not name.endswith("position_ids")}


def text_encoders(tensors):
    """``conditioner.embedders.*`` as the two towers ``text_encoder.py`` wants.

    Embedder 0 is a ``transformers`` CLIP saved whole, so it needs only its
    prefix removed. Embedder 1 is OpenCLIP's own module and needs translating.
    """
    towers = {"text_encoder": {}, "text_encoder_2": {}}
    prefixes = {"0.transformer.": "text_encoder", "1.model.": "text_encoder_2"}

    for name, tensor in tensors.items():
        for prefix, tower in prefixes.items():
            if name.startswith(prefix):
                towers[tower][name[len(prefix):]] = tensor
                break
        else:
            raise ValueError(f"unrecognized conditioner key: {name}")

    return {"text_encoder": _clip_l(towers["text_encoder"]),
            "text_encoder_2": _clip_g(towers["text_encoder_2"])}
