"""The handful of neural-network layers an SDXL VAE is built out of.

Everything here is expressed with the ~25 operators the bindings expose. The
three that need explaining:

* **GroupNorm** is ``mean_variance_normalization`` over a regrouped view of the
  tensor. DirectML normalizes the axes you name, so viewing ``[1, C, H, W]`` as
  ``[1, G, C/G, H*W]`` and normalizing axes 0, 2 and 3 leaves one mean and
  variance per group. The per-channel affine cannot ride along -- DirectML wants
  the scale and bias to be 1 along every normalized axis, and the channel axis is
  normalized here -- so it is a separate multiply and add.

* **Broadcasting** is a stride trick. DirectML reads a tensor through whatever
  strides its descriptor carries, and a stride of 0 makes an axis repeat. So a
  ``[1, C, 1, 1]`` bias becomes a ``[1, C, H, W]`` view with strides
  ``[C, 1, 0, 0]`` and no data is copied.

* **Transposing** is the same trick with non-zero strides: ``[1, C, H, W]`` read
  as ``[1, 1, H*W, C]`` with strides ``[.., .., 1, H*W]`` is the NCHW-to-tokens
  reshape that attention needs.
"""

import math

import numpy as np

import directml as dml

FLOAT32 = dml.TensorDataType.FLOAT32
UINT32 = dml.TensorDataType.UINT32

NUMPY_DTYPES = {
    FLOAT32: np.float32,
    dml.TensorDataType.FLOAT16: np.float16,
    UINT32: np.uint32,
    dml.TensorDataType.INT32: np.int32,
}


def sizes(expression):
    """The shape of an expression's output, as a list."""
    return list(expression.get_output_desc().sizes)


def data_type(expression):
    """The element type of an expression's output.

    Every reshape below reads this off the expression rather than assuming
    float32, so the same layer code builds a half-precision graph.
    """
    return expression.get_output_desc().data_type


class Model:
    """A graph under construction, plus the arrays bound to its inputs.

    Weights are registered with :meth:`constant` and carry their data with them.
    Anything fed per run -- the latent, the image -- is registered with
    :meth:`placeholder` and supplied to :meth:`run`.
    """

    def __init__(self, device, tensor_type=FLOAT32):
        self.tensor_type = tensor_type
        self.data_type = NUMPY_DTYPES[tensor_type]
        self.device = device
        self.graph = dml.GraphBuilder(device)
        self._bindings = []
        self._expressions = []
        # index -> (shape, dtype) for the inputs supplied per run.
        self._placeholders = {}
        self._outputs = []
        self._operator = None

    def _add_input(self, array, flags, data_type):
        desc = dml.TensorDesc(data_type, flags, list(array.shape))
        expression = dml.input_tensor(self.graph, len(self._bindings), desc)
        self._expressions.append(expression)
        # Bound here rather than at compile time so the caller can drop its own
        # reference to the array immediately. Holding every weight until the
        # whole graph is built costs a second copy of the model.
        self._bindings.append(dml.Binding(expression, array))
        return expression

    def constant(self, array, shape=None):
        """Register a weight, reshaped to ``shape`` if given.

        OWNED_BY_DML hands the tensor to DirectML at initialization, which is
        once per model, and it stays on the GPU from then on. Without the flag
        it would be re-uploaded on every dispatch.
        """
        array = np.ascontiguousarray(array, self.data_type)
        if shape is not None:
            array = array.reshape(shape)
        return self._add_input(array, dml.TensorFlags.OWNED_BY_DML, self.tensor_type)

    def placeholder(self, shape, data_type=None):
        """Register an input whose data is supplied to :meth:`run`.

        Token indices are the reason this takes a data type: ``gather`` wants an
        integer tensor, and a Binding now converts to whatever the tensor
        declares rather than forcing float32.
        """
        data_type = self.tensor_type if data_type is None else data_type
        expression = self._add_input(
            np.zeros(shape, NUMPY_DTYPES[data_type]), dml.TensorFlags.NONE, data_type)
        self._placeholders[len(self._bindings) - 1] = (list(shape), NUMPY_DTYPES[data_type])
        return expression

    def compile(self, outputs):
        self._outputs = list(outputs)
        self._operator = self.graph.build(dml.ExecutionFlags.NONE, self._outputs)
        # Uploads every OWNED_BY_DML tensor and hands it to DirectML. After this
        # a dispatch only carries the placeholders.
        self.device.initialize(self._operator, self._bindings)

        # DirectML holds the weights now, so each binding's copy can go. That is
        # 5.1 GiB for the UNet at half precision.
        for index, binding in enumerate(self._bindings):
            if index not in self._placeholders:
                binding.release_data()
        return self

    def run(self, *values):
        """Bind ``values`` to the placeholders in order and execute the graph."""
        if self._operator is None:
            raise RuntimeError("compile() the model before running it")
        if len(values) != len(self._placeholders):
            raise ValueError(f"expected {len(self._placeholders)} inputs, got {len(values)}")

        for (index, (shape, dtype)), value in zip(sorted(self._placeholders.items()), values):
            value = np.ascontiguousarray(value, dtype)
            if list(value.shape) != shape:
                raise ValueError(f"input {index} has shape {list(value.shape)}, expected {shape}")
            self._bindings[index] = dml.Binding(self._expressions[index], value)

        results = self.device.dispatch(self._operator, self._bindings, self._outputs)
        return [np.asarray(r).reshape(sizes(o)) for r, o in zip(results, self._outputs)]

    @property
    def input_count(self):
        return len(self._bindings)

    @property
    def temporary_size(self):
        """Scratch bytes one dispatch needs -- where the intermediates live."""
        return self._operator.temporary_size

    @property
    def persistent_size(self):
        """Bytes the weights occupy once DirectML has laid them out."""
        return self._operator.persistent_size


def broadcast(expression, shape):
    """View ``expression`` as ``shape``, repeating any axis whose extent is 1."""
    source = sizes(expression)
    if source == list(shape):
        return expression
    if len(source) != len(shape):
        raise ValueError(f"cannot broadcast {source} to {list(shape)}: rank differs")

    packed = [1] * len(source)
    for i in range(len(source) - 2, -1, -1):
        packed[i] = packed[i + 1] * source[i + 1]

    strides = []
    for size, target, stride in zip(source, shape, packed):
        if size == target:
            strides.append(stride)
        elif size == 1:
            strides.append(0)
        else:
            raise ValueError(f"cannot broadcast {source} to {list(shape)}: axis of {size}")

    return dml.reinterpret(expression, data_type(expression), list(shape), strides)


def to_tokens(expression):
    """View ``[1, C, H, W]`` as ``[1, 1, H*W, C]`` -- a transpose, not a copy."""
    n, c, h, w = sizes(expression)
    return dml.reinterpret(expression, data_type(expression), [n, 1, h * w, c],
                       [c * h * w, c * h * w, 1, h * w])


def to_image(expression, height, width):
    """The inverse of :func:`to_tokens`."""
    n, _, tokens, c = sizes(expression)
    if tokens != height * width:
        raise ValueError(f"{tokens} tokens do not fill {height}x{width}")
    return dml.reinterpret(expression, data_type(expression), [n, c, height, width],
                       [c * tokens, 1, width * c, c])


def silu(expression):
    """x * sigmoid(x), the activation the VAE calls "swish"."""
    return dml.multiply(expression, dml.activation_sigmoid(expression))


def conv2d(model, x, weight, bias, stride=1, padding=1, end_padding=None):
    """A 2-D convolution from torch-ordered ``[out, in, kh, kw]`` weights."""
    filters = model.constant(weight)
    biases = model.constant(bias, shape=[1, weight.shape[0], 1, 1])
    end = padding if end_padding is None else end_padding
    return dml.convolution(
        x, filters, biases,
        strides=[stride, stride],
        start_padding=[padding, padding],
        end_padding=[end, end])


def linear(model, x, weight, bias=None):
    """A dense layer over the last axis, from torch-ordered ``[out, in]`` weights.

    The UNet's attention projections carry no bias, so it is optional.
    """
    out_features = weight.shape[0]
    weights = model.constant(weight, shape=[1, 1, out_features, weight.shape[1]])
    transpose = dml.MatrixTransform.TRANSPOSE

    # A gemm wants both operands to agree on the batch axes, and a weight has
    # none of its own, so it is repeated across the batch at a stride of zero.
    batch = sizes(x)[0]
    if batch != 1:
        weights = broadcast(weights, [batch, 1, out_features, weight.shape[1]])
    if bias is None:
        return dml.gemm(x, weights, trans_b=transpose)

    biases = model.constant(bias, shape=[1, 1, 1, out_features])
    shape = sizes(x)[:-1] + [out_features]
    return dml.gemm(x, weights, broadcast(biases, shape), trans_b=transpose)


def group_norm(model, x, weight, bias, groups=32, epsilon=1e-6):
    """GroupNorm over ``[1, C, H, W]``, affine applied per channel afterwards."""
    n, c, h, w = shape = sizes(x)
    if c % groups:
        raise ValueError(f"{c} channels do not divide into {groups} groups")

    grouped = dml.reinterpret(x, data_type(x), [n, groups, c // groups, h * w], None)
    normalized = dml.mean_variance_normalization(
        grouped, None, None, [2, 3],
        normalize_variance=True, normalize_mean=True, epsilon=epsilon)
    normalized = dml.reinterpret(normalized, data_type(x), shape, None)

    scale = broadcast(model.constant(weight, shape=[1, c, 1, 1]), shape)
    shift = broadcast(model.constant(bias, shape=[1, c, 1, 1]), shape)
    return dml.add(dml.multiply(normalized, scale), shift)


def resnet_block(model, x, params, prefix, epsilon=1e-6):
    """The VAE's ResnetBlock2D: two norm/SiLU/conv pairs plus a residual."""
    h = conv2d(model, silu(group_norm(model, x, params[f"{prefix}.norm1.weight"],
                                      params[f"{prefix}.norm1.bias"], epsilon=epsilon)),
               params[f"{prefix}.conv1.weight"], params[f"{prefix}.conv1.bias"])
    h = conv2d(model, silu(group_norm(model, h, params[f"{prefix}.norm2.weight"],
                                      params[f"{prefix}.norm2.bias"], epsilon=epsilon)),
               params[f"{prefix}.conv2.weight"], params[f"{prefix}.conv2.bias"])

    # The shortcut is a 1x1 convolution only when the block changes width.
    if f"{prefix}.conv_shortcut.weight" in params:
        x = conv2d(model, x, params[f"{prefix}.conv_shortcut.weight"],
                   params[f"{prefix}.conv_shortcut.bias"], padding=0)
    return dml.add(x, h)


def attention_block(model, x, params, prefix, epsilon=1e-6):
    """Single-head self-attention over every pixel, with a residual.

    The projections run in token layout so the whole block is four matrix
    multiplies and a softmax; the only reshapes are stride tricks.
    """
    _, channels, height, width = sizes(x)

    normalized = group_norm(model, x, params[f"{prefix}.group_norm.weight"],
                            params[f"{prefix}.group_norm.bias"], epsilon=epsilon)
    tokens = to_tokens(normalized)

    query = linear(model, tokens, params[f"{prefix}.to_q.weight"], params[f"{prefix}.to_q.bias"])
    key = linear(model, tokens, params[f"{prefix}.to_k.weight"], params[f"{prefix}.to_k.bias"])
    value = linear(model, tokens, params[f"{prefix}.to_v.weight"], params[f"{prefix}.to_v.bias"])

    scores = dml.gemm(query, key, trans_b=dml.MatrixTransform.TRANSPOSE,
                      alpha=1.0 / math.sqrt(channels))
    attended = dml.gemm(dml.activation_softmax(scores, [3]), value)

    projected = linear(model, attended, params[f"{prefix}.to_out.0.weight"],
                       params[f"{prefix}.to_out.0.bias"])
    return dml.add(to_image(projected, height, width), x)


def upsample_nearest(x, scale=2):
    """Nearest-neighbour upsampling, the only interpolation the VAE uses."""
    return dml.up_sample_2d(x, dml.Size2D(scale, scale),
                            dml.InterpolationMode.NEAREST_NEIGHBOR)


def crop_to(x, height, width):
    """Trim an image tensor to ``height`` by ``width``, from the top left.

    A level that halved an odd extent on the way down cannot get back to it by
    doubling on the way up, so the up path can come out a pixel wider or taller
    than the skip connection it has to be concatenated with. Cropping a
    nearest-neighbour 2x upsample to ``2n - 1`` is exactly a nearest-neighbour
    resize to ``2n - 1`` -- the two pick the same source pixel for every
    destination -- which is what diffusers does by handing the upsampler the
    size it is aiming for.
    """
    batch, channels, current_height, current_width = sizes(x)
    if (current_height, current_width) == (height, width):
        return x
    return dml.slice(x, [0, 0, 0, 0], [batch, channels, height, width], [1, 1, 1, 1])


# --- Transformer pieces, used by the text encoders -------------------------


def layer_norm(model, x, weight, bias, epsilon=1e-5):
    """LayerNorm over the last axis of a token tensor.

    Same shape rule as :func:`group_norm`: the affine cannot ride along on the
    normalization, because the axis it varies over is the axis being normalized.
    """
    shape = sizes(x)
    leading = [1] * (len(shape) - 1)
    normalized = dml.mean_variance_normalization(
        x, None, None, [len(shape) - 1],
        normalize_variance=True, normalize_mean=True, epsilon=epsilon)

    scale = broadcast(model.constant(weight, shape=leading + [shape[-1]]), shape)
    shift = broadcast(model.constant(bias, shape=leading + [shape[-1]]), shape)
    return dml.add(dml.multiply(normalized, scale), shift)


def quick_gelu(x):
    """x * sigmoid(1.702x), the approximation CLIP ViT-L was trained with."""
    return dml.multiply(x, dml.activation_sigmoid(dml.activation_linear(x, 1.702, 0.0)))


def split_heads(x, heads):
    """View ``[1, 1, T, C]`` as ``[1, heads, T, C/heads]``. No copy."""
    n, _, tokens, channels = sizes(x)
    if channels % heads:
        raise ValueError(f"{channels} channels do not divide into {heads} heads")
    dim = channels // heads
    return dml.reinterpret(x, data_type(x), [n, heads, tokens, dim],
                           [tokens * channels, dim, channels, 1])


def merge_heads(x):
    """The inverse of :func:`split_heads`, which costs one copy.

    Strides cannot express this one. Going from ``[1, heads, T, dim]`` back to
    ``[1, 1, T, heads*dim]`` means an output channel index splits into a head and
    an offset within it, and an offset that divides an index is not a stride. So
    view the buffer as ``[1, T, heads, dim]`` -- that much *is* a stride trick --
    then let an identity operator write it out packed, after which the last
    reshape is free.
    """
    n, heads, tokens, dim = sizes(x)
    transposed = dml.reinterpret(x, data_type(x), [n, tokens, heads, dim],
                                 [heads * tokens * dim, dim, tokens * dim, 1])
    packed = dml.activation_identity(transposed)
    return dml.reinterpret(packed, data_type(x), [n, 1, tokens, heads * dim], None)


def attend(query, key, value, heads, mask=None):
    """Scaled dot-product attention over already-projected token tensors.

    Key and value may carry a different number of tokens than the query, which
    is what makes this cross-attention as well as self-attention.
    """
    dim = sizes(query)[-1] // heads
    if mask is None:
        # One operator instead of three. The score matrix stays inside it rather
        # than becoming a tensor the graph has to find room for twice.
        return dml.multihead_attention(query, key, value, heads, 1.0 / math.sqrt(dim))

    # DML_MULTIHEAD_ATTENTION_MASK_TYPE has no additive mask, and CLIP's is
    # additive and causal, so a masked attention is still written out by hand.
    query, key, value = (split_heads(t, heads) for t in (query, key, value))

    scores = dml.gemm(query, key, trans_b=dml.MatrixTransform.TRANSPOSE,
                      alpha=1.0 / math.sqrt(dim))
    scores = dml.add(scores, broadcast(mask, sizes(scores)))

    return merge_heads(dml.gemm(dml.activation_softmax(scores, [3]), value))


def multi_head_attention(model, x, params, prefix, heads, mask=None):
    """Self-attention under CLIP's weight names, where every projection has a bias."""
    def project(name):
        return linear(model, x, params[f"{prefix}.{name}.weight"], params[f"{prefix}.{name}.bias"])

    attended = attend(project("q_proj"), project("k_proj"), project("v_proj"), heads, mask)
    return linear(model, attended, params[f"{prefix}.out_proj.weight"],
                  params[f"{prefix}.out_proj.bias"])


def diffusers_attention(model, x, params, prefix, heads, context=None):
    """Attention under diffusers' weight names, where only the output projects a bias.

    Passing ``context`` makes it cross-attention: the query still comes from the
    image tokens, the keys and values from the text embeddings.
    """
    source = x if context is None else context
    query = linear(model, x, params[f"{prefix}.to_q.weight"])
    key = linear(model, source, params[f"{prefix}.to_k.weight"])
    value = linear(model, source, params[f"{prefix}.to_v.weight"])

    attended = attend(query, key, value, heads)
    return linear(model, attended, params[f"{prefix}.to_out.0.weight"],
                  params[f"{prefix}.to_out.0.bias"])


def to_channels(expression):
    """View ``[1, 1, 1, C]`` as ``[1, C, 1, 1]``, ready to broadcast over an image."""
    n, _, _, c = sizes(expression)
    return dml.reinterpret(expression, data_type(expression), [n, c, 1, 1], None)


def geglu(model, x, params, prefix):
    """The UNet's feed-forward: project to twice the width, gate one half by the other."""
    projected = linear(model, x, params[f"{prefix}.net.0.proj.weight"],
                       params[f"{prefix}.net.0.proj.bias"])
    n, _, tokens, doubled = sizes(projected)
    inner = doubled // 2

    def half(offset):
        return dml.slice(projected, [0, 0, 0, offset], [n, 1, tokens, inner], [1, 1, 1, 1])

    gated = dml.multiply(half(0), dml.activation_gelu(half(inner)))
    return linear(model, gated, params[f"{prefix}.net.2.weight"], params[f"{prefix}.net.2.bias"])
