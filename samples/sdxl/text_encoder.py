"""SDXL's two CLIP text encoders, built as DirectML graphs.

SDXL conditions on two text towers at once: CLIP ViT-L/14 (768 wide, 12 layers)
and OpenCLIP ViT-bigG/14 (1280 wide, 32 layers). Their per-token outputs are
concatenated into the 2048-wide sequence the UNet cross-attends to, and bigG
additionally contributes a single pooled vector that feeds the UNet's timestep
conditioning.

Two details that are easy to get wrong and that nothing will complain about:

* The embeddings SDXL uses are the **penultimate** layer's output, not the last
  one's, and the final layer norm is not applied to them. It is applied to the
  last layer's output, which is where the pooled vector comes from.
* The two towers use different activations. ViT-L was trained with QuickGELU,
  ``x * sigmoid(1.702x)``; bigG uses real GELU.

The tokenizer is `transformers.CLIPTokenizer` -- BPE over a 49408-entry vocab is
string processing, not a DirectML concern, and transformers ships it without
needing PyTorch.
"""

import numpy as np

import directml as dml

from dml_layers import (
    Model, layer_norm, linear, multi_head_attention, quick_gelu)

MAX_TOKENS = 77
LAYER_NORM_EPSILON = 1e-5

# subfolder -> the tower's shape. `projection` is set only for the tower whose
# pooled output SDXL uses, which is the one saved as CLIPTextModelWithProjection.
CONFIGS = {
    "text_encoder": dict(width=768, layers=12, heads=12,
                         activation="quick_gelu", projection=None),
    "text_encoder_2": dict(width=1280, layers=32, heads=20,
                           activation="gelu", projection=1280),
}


def causal_mask(tokens=MAX_TOKENS):
    """Additive mask that stops a token from attending to later ones."""
    # float32's most negative value is what CLIP is evaluated with; it survives
    # the softmax's max-subtraction without rounding to -inf.
    return np.triu(np.full((1, 1, tokens, tokens), np.finfo(np.float32).min,
                           dtype=np.float32), k=1)


def _encoder_layer(model, x, params, prefix, heads, activation, mask):
    """One pre-norm transformer block: attention, then MLP, both residual."""
    x = dml.add(x, multi_head_attention(
        model,
        layer_norm(model, x, params[f"{prefix}.layer_norm1.weight"],
                   params[f"{prefix}.layer_norm1.bias"], LAYER_NORM_EPSILON),
        params, f"{prefix}.self_attn", heads, mask))

    h = layer_norm(model, x, params[f"{prefix}.layer_norm2.weight"],
                   params[f"{prefix}.layer_norm2.bias"], LAYER_NORM_EPSILON)
    h = activation(linear(model, h, params[f"{prefix}.mlp.fc1.weight"],
                          params[f"{prefix}.mlp.fc1.bias"]))
    h = linear(model, h, params[f"{prefix}.mlp.fc2.weight"], params[f"{prefix}.mlp.fc2.bias"])
    return dml.add(x, h)


def build_text_encoder(model, params, config):
    """Add a text tower to ``model``. Returns (token input, list of outputs).

    The outputs are the penultimate hidden state, and -- for the tower with a
    projection -- the final hidden state after its layer norm, from which the
    caller pools.
    """
    width, layers, heads = config["width"], config["layers"], config["heads"]
    activation = quick_gelu if config["activation"] == "quick_gelu" else dml.activation_gelu
    wants_pooled = config["projection"] is not None

    table = params["text_model.embeddings.token_embedding.weight"]
    tokens = model.placeholder([1, 1, 1, MAX_TOKENS], np.uint32)
    x = dml.gather(model.constant(table, shape=[1, 1, table.shape[0], width]),
                   tokens, axis=2, index_dimensions=1)
    x = dml.add(x, model.constant(params["text_model.embeddings.position_embedding.weight"],
                                  shape=[1, 1, MAX_TOKENS, width]))

    mask = model.constant(causal_mask())

    # The last layer only earns its keep when the pooled vector is wanted; the
    # sequence SDXL cross-attends to comes from the one before it.
    penultimate = None
    for i in range(layers if wants_pooled else layers - 1):
        x = _encoder_layer(model, x, params, f"text_model.encoder.layers.{i}",
                           heads, activation, mask)
        if i == layers - 2:
            penultimate = x

    if not wants_pooled:
        return tokens, [penultimate]

    final = layer_norm(model, x, params["text_model.final_layer_norm.weight"],
                       params["text_model.final_layer_norm.bias"], LAYER_NORM_EPSILON)
    return tokens, [penultimate, final]


def text_encoder(device, params, config):
    """Compile one text tower."""
    model = Model(device)
    _, outputs = build_text_encoder(model, params, config)
    return model.compile(outputs)


class TextEncoders:
    """Both towers plus their tokenizers: prompt in, conditioning out."""

    def __init__(self, device, weights, tokenizers):
        self.tokenizers = tokenizers
        self.models = {name: text_encoder(device, weights[name], CONFIGS[name])
                       for name in CONFIGS}
        # nn.Linear(bias=False), applied on the CPU to a single pooled vector.
        self.projection = weights["text_encoder_2"]["text_projection.weight"]

    def encode(self, prompt):
        """Return (prompt_embeds [77, 2048], pooled_embeds [1280])."""
        sequences, pooled = [], None

        for name, config in CONFIGS.items():
            ids = np.array(self.tokenizers[name](
                prompt, padding="max_length", max_length=MAX_TOKENS,
                truncation=True)["input_ids"], np.uint32)

            outputs = self.models[name].run(ids.reshape(1, 1, 1, MAX_TOKENS))
            sequences.append(outputs[0].reshape(MAX_TOKENS, config["width"]))

            if config["projection"] is not None:
                # CLIP pools at the end-of-text token, which is the highest id
                # in the vocabulary and so the first one argmax finds.
                end_of_text = int(np.argmax(ids))
                final = outputs[1].reshape(MAX_TOKENS, config["width"])
                pooled = final[end_of_text] @ self.projection.T

        return np.concatenate(sequences, axis=-1), pooled
