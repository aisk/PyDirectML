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


def sizes(expression):
    """The shape of an expression's output, as a list."""
    return list(expression.get_output_desc().sizes)


class Model:
    """A graph under construction, plus the arrays bound to its inputs.

    Weights are registered with :meth:`constant` and carry their data with them.
    Anything fed per run -- the latent, the image -- is registered with
    :meth:`placeholder` and supplied to :meth:`run`.
    """

    def __init__(self, device):
        self.device = device
        self.graph = dml.GraphBuilder(device)
        self._expressions = []
        self._arrays = []
        self._placeholders = []
        self._bindings = []
        self._outputs = []
        self._operator = None

    def _add_input(self, shape, array, flags):
        desc = dml.TensorDesc(FLOAT32, flags, list(shape))
        expression = dml.input_tensor(self.graph, len(self._expressions), desc)
        self._expressions.append(expression)
        self._arrays.append(array)
        return expression

    def constant(self, array, shape=None):
        """Register a weight, reshaped to ``shape`` if given.

        OWNED_BY_DML hands the tensor to DirectML at initialization, which is
        once per model, and it stays on the GPU from then on. Without the flag
        it would be re-uploaded on every dispatch.
        """
        array = np.ascontiguousarray(array, np.float32)
        if shape is not None:
            array = array.reshape(shape)
        return self._add_input(array.shape, array, dml.TensorFlags.OWNED_BY_DML)

    def placeholder(self, shape):
        """Register an input whose data is supplied to :meth:`run`."""
        expression = self._add_input(shape, np.zeros(shape, np.float32), dml.TensorFlags.NONE)
        self._placeholders.append(len(self._expressions) - 1)
        return expression

    def compile(self, outputs):
        self._outputs = list(outputs)
        self._operator = self.graph.build(dml.ExecutionFlags.NONE, self._outputs)
        # Binding copies the array into an upload buffer, so the weights are
        # copied once here rather than on every run.
        self._bindings = [dml.Binding(e, a) for e, a in zip(self._expressions, self._arrays)]
        # Uploads every OWNED_BY_DML tensor and hands it to DirectML. After this
        # a dispatch only carries the placeholders.
        self.device.initialize(self._operator, self._bindings)
        return self

    def run(self, *values):
        """Bind ``values`` to the placeholders in order and execute the graph."""
        if self._operator is None:
            raise RuntimeError("compile() the model before running it")
        if len(values) != len(self._placeholders):
            raise ValueError(f"expected {len(self._placeholders)} inputs, got {len(values)}")

        for index, value in zip(self._placeholders, values):
            expected = list(self._arrays[index].shape)
            value = np.ascontiguousarray(value, np.float32)
            if list(value.shape) != expected:
                raise ValueError(f"input {index} has shape {list(value.shape)}, expected {expected}")
            self._bindings[index] = dml.Binding(self._expressions[index], value)

        results = self.device.dispatch(self._operator, self._bindings, self._outputs)
        return [np.array(r, np.float32).reshape(sizes(o)) for r, o in zip(results, self._outputs)]

    @property
    def input_count(self):
        return len(self._expressions)


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

    return dml.reinterpret(expression, FLOAT32, list(shape), strides)


def to_tokens(expression):
    """View ``[1, C, H, W]`` as ``[1, 1, H*W, C]`` -- a transpose, not a copy."""
    n, c, h, w = sizes(expression)
    return dml.reinterpret(expression, FLOAT32, [n, 1, h * w, c], [c * h * w, c * h * w, 1, h * w])


def to_image(expression, height, width):
    """The inverse of :func:`to_tokens`."""
    n, _, tokens, c = sizes(expression)
    if tokens != height * width:
        raise ValueError(f"{tokens} tokens do not fill {height}x{width}")
    return dml.reinterpret(expression, FLOAT32, [n, c, height, width], [c * tokens, 1, width * c, c])


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


def linear(model, x, weight, bias):
    """A dense layer over the last axis, from torch-ordered ``[out, in]`` weights."""
    out_features = weight.shape[0]
    weights = model.constant(weight, shape=[1, 1, out_features, weight.shape[1]])
    biases = model.constant(bias, shape=[1, 1, 1, out_features])
    shape = sizes(x)[:-1] + [out_features]
    return dml.gemm(
        x, weights, broadcast(biases, shape),
        trans_b=dml.MatrixTransform.TRANSPOSE)


def group_norm(model, x, weight, bias, groups=32, epsilon=1e-6):
    """GroupNorm over ``[1, C, H, W]``, affine applied per channel afterwards."""
    n, c, h, w = shape = sizes(x)
    if c % groups:
        raise ValueError(f"{c} channels do not divide into {groups} groups")

    grouped = dml.reinterpret(x, FLOAT32, [n, groups, c // groups, h * w], None)
    normalized = dml.mean_variance_normalization(
        grouped, None, None, [2, 3],
        normalize_variance=True, normalize_mean=True, epsilon=epsilon)
    normalized = dml.reinterpret(normalized, FLOAT32, shape, None)

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
