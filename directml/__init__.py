"""Python binding for DirectML.

The package splits in two along the line drawn in docs/api-design.md: ``_core``
is the compiled extension and owns resources, execution and the data hot path;
this wrapper layer owns signature shaping, defaults, validation and error
messages. The boundary of the library is ``import directml``, not the ``.pyd``.

Every class here is the ``_core`` class itself, with the wrapper's methods and
properties attached at import time. Instances are created on the C++ side --
an Expression by an operator, a Buffer by dispatch -- so subclassing would
give the wrapper's conveniences to the objects users construct and not to the
ones the library hands back. Attaching gives them to both.
"""

import importlib.metadata
import math
import typing

import numpy as np

from . import _core
from ._core import *  # noqa: F401,F403 -- the classes, enums and operators

try:
    __version__ = importlib.metadata.version("directml")
except importlib.metadata.PackageNotFoundError:
    # Running from a source tree that was never pip-installed.
    __version__ = "0.0.0"


# How a DML_TENSOR_DATA_TYPE is spelled in numpy. Every sample used to carry a
# private copy of this table; it lives here now, and every API that takes a
# data type accepts either spelling.
_NUMPY_DTYPES = {
    TensorDataType.FLOAT32: np.dtype(np.float32),
    TensorDataType.FLOAT16: np.dtype(np.float16),
    TensorDataType.UINT32: np.dtype(np.uint32),
    TensorDataType.UINT16: np.dtype(np.uint16),
    TensorDataType.UINT8: np.dtype(np.uint8),
    TensorDataType.INT32: np.dtype(np.int32),
    TensorDataType.INT16: np.dtype(np.int16),
    TensorDataType.INT8: np.dtype(np.int8),
    TensorDataType.FLOAT64: np.dtype(np.float64),
    TensorDataType.UINT64: np.dtype(np.uint64),
    TensorDataType.INT64: np.dtype(np.int64),
}

_TENSOR_DATA_TYPES = {dtype: data_type for data_type, dtype in _NUMPY_DTYPES.items()}


def _to_data_type(dtype):
    """The TensorDataType for ``dtype``, which may be either spelling."""
    if isinstance(dtype, TensorDataType):
        return dtype
    try:
        return _TENSOR_DATA_TYPES[np.dtype(dtype)]
    except (KeyError, TypeError):
        raise TypeError(f"{dtype!r} is not a tensor data type DirectML supports") from None


# --- TensorDesc: numpy dtypes in, list arguments normalized -----------------


_core_tensor_desc_init = _core.TensorDesc.__init__


def _tensor_desc_init(self, data_type, sizes, *, flags=TensorFlags.NONE, strides=None,
                      total_tensor_size_in_bytes=None, guaranteed_base_offset_alignment=0,
                      tensor_policy=None):
    """A tensor's element type, shape, strides and flags.

    ``graph.input(...)`` covers the common cases; this is for precise control
    over strides, total size and alignment.

    Args:
        data_type: A numpy dtype or TensorDataType.
        sizes: The tensor's shape.
        flags: TensorFlags; OWNED_BY_DML marks a tensor handed over at
            initialize().
        strides: Element strides for a non-packed view. Exclusive with
            tensor_policy, which decides the strides itself.
        total_tensor_size_in_bytes: Override the size DirectML computes from
            the sizes and strides.
        guaranteed_base_offset_alignment: Alignment the caller promises for
            the tensor's offset within its buffer.
        tensor_policy: A TensorPolicy deciding the layout.
    """
    _core_tensor_desc_init(
        self, _to_data_type(data_type), list(sizes), flags=flags,
        strides=None if strides is None else list(strides),
        total_tensor_size_in_bytes=total_tensor_size_in_bytes,
        guaranteed_base_offset_alignment=guaranteed_base_offset_alignment,
        tensor_policy=tensor_policy)


_core.TensorDesc.__init__ = _tensor_desc_init


# --- Expression and Buffer conveniences -------------------------------------
# Both are read through a TensorDesc, so both get the same view of it.


def _desc_shape(self):
    return tuple(self.desc.sizes)


def _desc_strides(self):
    strides = self.desc.strides
    return None if strides is None else tuple(strides)


def _desc_dtype(self):
    return _NUMPY_DTYPES[self.desc.data_type]


def _desc_size(self):
    return math.prod(self.desc.sizes)


def _desc_repr(name):
    def __repr__(self):
        desc = self.desc
        return f"<dml.{name} {_NUMPY_DTYPES[desc.data_type].name} {list(desc.sizes)}>"
    return __repr__


for _class, _name in [(_core.Expression, "Expression"), (_core.Buffer, "Buffer")]:
    _class.shape = property(
        _desc_shape, doc=f"The shape of this {_name}, as a tuple.")
    _class.strides = property(
        _desc_strides, doc=f"The element strides of this {_name}, or None if packed.")
    _class.dtype = property(
        _desc_dtype, doc=f"The element type of this {_name}, as a numpy dtype.")
    _class.size = property(
        _desc_size, doc=f"The number of elements in this {_name}.")
    _class.__repr__ = _desc_repr(_name)
del _class, _name


_core_buffer_init = _core.Buffer.__init__


def _buffer_init(self, device, array, dtype=None):
    """Upload an array to the GPU.

    A Buffer is a tensor that lives in the device's memory. dispatch() accepts
    one wherever it accepts an array, and binds it in place with no upload;
    ``dispatch(readback=False)`` returns one per output instead of copying the
    outputs back. That is how a tensor moves between two graphs without
    crossing PCIe twice.

    Args:
        device: The Device whose memory to use.
        array: The data; anything np.asarray accepts.
        dtype: The buffer's element type. Defaults to the array's own dtype.
            The array is converted under the same rules as dispatch(): within
            a dtype kind or NumPy-safe silently, anything else must be an
            explicit astype().

    Raises:
        ValueError: The array's dtype would convert unsafely.
        TypeError: The dtype is not one DirectML supports.
    """
    array = np.asarray(array)
    target = _NUMPY_DTYPES[_to_data_type(array.dtype if dtype is None else dtype)]
    _check_cast(array, target, "the buffer")
    desc = _core.TensorDesc(target, list(array.shape) or [1])
    _core_buffer_init(self, device, desc, np.ascontiguousarray(array, target))


_core.Buffer.__init__ = _buffer_init


# --- Binding rules -----------------------------------------------------------


def _check_cast(array, target, what):
    """Refuse a conversion that would silently lose meaning.

    Narrowing within a kind is fine -- float64 to float32 is what np.zeros()
    lands on, float32 to float16 is how half-precision weights get loaded --
    and so is any cast NumPy calls safe. Crossing kinds any other way, int32
    into a float32 tensor above all, is the silent-wrong-answer case, and has
    to be spelled out with an explicit astype. NumPy's own 'same_kind' rule is
    no help here: it permits int32 to float32.
    """
    if (array.dtype != target and array.dtype.kind != target.kind
            and not np.can_cast(array.dtype, target, "safe")):
        raise ValueError(
            f"cannot convert a {array.dtype} array to {target.name} for {what}; "
            f"convert it explicitly with astype()")


def _check_fit(array, dtype, sizes, total_bytes, strided, what):
    """Refuse an array or Buffer that does not fill its tensor."""
    if strided:
        # A strided view repeats or reorders elements, so the data covers the
        # underlying buffer rather than the logical shape.
        nbytes = array.nbytes if isinstance(array, _core.Buffer) else array.size * dtype.itemsize
        if nbytes < total_bytes:
            raise ValueError(
                f"{nbytes} bytes do not fill {what}, whose buffer holds {total_bytes} bytes")
    elif array.size != math.prod(sizes):
        kind = "Buffer" if isinstance(array, _core.Buffer) else "array"
        raise ValueError(f"{kind} of {array.size} elements does not fill {what}")


class _Slot:
    """What the wrapper needs to know about one graph input."""

    __slots__ = ("index", "owned", "name", "dtype", "sizes", "total_bytes", "strided")

    def __init__(self, index, owned, desc, name):
        self.index = index
        self.owned = owned
        self.name = name
        self.dtype = _NUMPY_DTYPES[desc.data_type]
        self.sizes = tuple(desc.sizes)
        self.total_bytes = desc.total_tensor_size_in_bytes
        self.strided = desc.strides is not None

    def describe(self):
        name = f" {self.name!r}" if self.name is not None else ""
        return f"input {self.index}{name} ({self.dtype.name} {list(self.sizes)})"


class _Slots:
    """A compiled operator's inputs, by index, node identity and name."""

    def __init__(self, op):
        names = getattr(op, "_names", {})
        slots = op._input_slots
        self.keys = [key for key, _, _ in slots]
        self.by_index = [
            _Slot(index, owned, desc, names.get(key))
            for index, (key, owned, desc) in enumerate(slots)
        ]
        self.by_node = dict(zip(self.keys, self.by_index))
        self.by_name = {slot.name: slot for slot in self.by_index if slot.name is not None}

    def lookup(self, key, verb):
        if isinstance(key, _core.Expression):
            slot = self.by_node.get(key._node_id)
            if slot is None:
                raise ValueError(f"{key!r} is not an input of this graph")
            return slot
        if isinstance(key, str):
            slot = self.by_name.get(key)
            if slot is None:
                raise ValueError(f"this graph has no input named {key!r}")
            return slot
        raise TypeError(
            f"{verb}() keys must be Expressions or input names, not {type(key).__name__}")

    def owned(self):
        return [slot for slot in self.by_index if slot.owned]


def _slot_table(op):
    table = getattr(op, "_slot_cache", None)
    if table is None:
        table = op._slot_cache = _Slots(op)
    return table


def _validate(op, inputs, owned, verb, defaults=None):
    """Match the dict against the graph's inputs.

    Returns ``(buffers, staged)``: the inputs supplied as Buffers, bound in
    place, and the arrays to convert and upload, both by input index.
    Everything that can go wrong is raised from here, before any upload
    starts: a key that is not an input, one bound in the wrong phase or twice,
    a dtype that would silently lose meaning, data of the wrong size, and
    inputs that are missing outright. ``defaults`` supplies the constants the
    graph recorded, for any owned input the dict leaves out.
    """
    table = _slot_table(op)
    buffers = {}
    staged = {}

    def bind(slot, value):
        if slot.owned != owned:
            if owned:
                raise ValueError(
                    f"{slot.describe()} is not owned by DirectML; bind it at dispatch instead")
            raise ValueError(
                f"{slot.describe()} is owned by DirectML; bind it in initialize() instead")
        if slot.index in buffers or slot.index in staged:
            raise ValueError(f"{slot.describe()} is bound twice")

        if isinstance(value, _core.Buffer):
            # A Buffer is already typed; there is nothing to convert it with.
            if value.dtype != slot.dtype:
                raise ValueError(
                    f"cannot bind a {value.dtype} Buffer to {slot.describe()}; "
                    f"Buffers are not converted")
            _check_fit(value, slot.dtype, slot.sizes, slot.total_bytes, slot.strided,
                       slot.describe())
            buffers[slot.index] = value
            return

        array = np.asarray(value)
        _check_cast(array, slot.dtype, slot.describe())
        _check_fit(array, slot.dtype, slot.sizes, slot.total_bytes, slot.strided,
                   slot.describe())
        staged[slot.index] = array

    for key, value in inputs.items():
        bind(table.lookup(key, verb), value)

    if defaults:
        for slot in table.by_index:
            if slot.index not in buffers and slot.index not in staged:
                key = table.keys[slot.index]
                if key in defaults:
                    bind(slot, defaults[key])

    missing = [
        slot for slot in table.by_index
        if slot.owned == owned and slot.index not in buffers and slot.index not in staged
    ]
    if missing:
        raise ValueError(
            f"{verb}() is missing " + ", ".join(slot.describe() for slot in missing))

    return buffers, staged


def _converted(op, staged):
    """Yield (index, contiguous array of the tensor's dtype) pairs.

    Conversion happens one tensor at a time, as the upload loop in _core pulls
    on this generator: converting the whole dict first would hold a second copy
    of every weight at once, which for model-sized graphs is gigabytes.
    """
    slots = _slot_table(op).by_index
    for index, array in staged.items():
        yield index, np.ascontiguousarray(array, slots[index].dtype)


# --- CompiledOperator: dict-shaped initialize and dispatch -------------------


def _operator_initialize(self, weights=None):
    """Upload the owned inputs and run the operator initializer.

    The data is read once, here, and lives on the GPU from then on; the
    library keeps no copy. To change an owned input's data afterwards, call
    initialize() again -- the previous contents are replaced wholesale, and
    every owned input must be supplied again, constants included.

    Args:
        weights: A dict mapping owned inputs to their data, keyed by
            Expression or by the name given at ``graph.input()``. A value is
            an array, converted to the tensor's dtype under the same rules as
            dispatch(), or a Buffer, bound as it is. Inputs declared with
            ``graph.constant()`` may be left out the first time: the graph
            recorded their data, and compile() handed it over.

    Raises:
        ValueError: A key is not an input of this graph, is not owned, or is
            bound twice; an owned input is missing; an array's dtype would
            convert unsafely (cross-kind and not NumPy-safe); a Buffer's
            dtype differs from the tensor's; or the data does not fit.
    """
    constants = getattr(self, "_constants", None)
    buffers, staged = _validate(self, weights or {}, owned=True, verb="initialize",
                                defaults=constants)
    self._initialize(buffers, _converted(self, staged))
    # The library keeps no copy: once the constants are on the GPU, the
    # references the graph recorded are dropped.
    self._constants = None


def _operator_dispatch(self, inputs=None, *, readback=True):
    """Run the operator. Calling the operator is the same as dispatching it.

    Args:
        inputs: A dict mapping every non-owned input to its data, keyed by
            Expression or by the name given at ``graph.input()``. An array is
            converted to its tensor's dtype if the conversion stays within a
            dtype kind or NumPy calls it safe; anything else must be an
            explicit astype(). A packed array short of DirectML's 4-byte
            rounding is padded. A Buffer is bound where it is, with no
            upload; its dtype must match the tensor's.
        readback: True copies each output back to the host as a numpy array
            of the output's shape and dtype. False leaves each output on the
            GPU and returns a Buffer for it, which is what to pass to the next
            graph.

    Returns:
        One array or Buffer per output of compile(), in that order.

    Raises:
        ValueError: The operator has an owned input that was never
            initialized, a key is not an input of this graph or is owned, an
            input is missing or bound twice, a dtype would convert unsafely,
            or the data does not fit its tensor.
    """
    if not self.initialized:
        raise ValueError(
            "initialize() this operator before dispatching it; it owns " +
            ", ".join(slot.describe() for slot in _slot_table(self).owned()))

    buffers, staged = _validate(self, inputs or {}, owned=False, verb="dispatch")
    return self._dispatch(buffers, _converted(self, staged), readback)


_core.CompiledOperator.initialize = _operator_initialize
_core.CompiledOperator.dispatch = _operator_dispatch
_core.CompiledOperator.__call__ = _operator_dispatch


# --- Graph: input, constant and compile ---------------------------------------


def _graph_register(self, expression, name):
    """Record an input's name on the graph, refusing a duplicate."""
    names = self.__dict__.setdefault("_names", {})
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(f"an input's name must be a str, not {type(name).__name__}")
        if name in names.values():
            raise ValueError(f"this graph already has an input named {name!r}")
        names[expression._node_id] = name
    return expression


def _graph_input(self, sizes=None, dtype=None, *, owned=False, strides=None, desc=None,
                 name=None):
    """Add an input tensor to the graph and return its Expression.

    The graph assigns the input's index; no caller ever writes one.

    Args:
        sizes: The tensor's shape.
        dtype: A numpy dtype or TensorDataType. Defaults to float32.
        owned: Mark the tensor DML_TENSOR_FLAG_OWNED_BY_DML: its data is
            handed over once, at initialize(), and lives on the GPU from
            then on. Without it the data rides along with every dispatch.
            For a weight whose value is already in hand, ``constant()``
            declares an owned input and records the data in one call.
        strides: Element strides for a non-packed view; a stride of 0 repeats
            an axis, which is how a broadcast input is declared.
        desc: A complete TensorDesc, for the controls input() does not take
            (total_tensor_size_in_bytes, guaranteed_base_offset_alignment).
            Every other tensor argument is illegal with it -- a desc already
            answers them, and answering twice would raise the question of
            which wins.
        name: An optional name, unique within the graph, that initialize()
            and dispatch() accept as a dict key in place of the Expression.

    Returns:
        The input's Expression, which is also its binding key at
        initialize() and dispatch().

    Raises:
        TypeError: Neither sizes nor desc was given, or desc was combined
            with another tensor argument.
        ValueError: The name is already taken.
    """
    if desc is not None:
        if sizes is not None or dtype is not None or owned or strides is not None:
            raise TypeError("desc= already describes the tensor; no other argument may be passed with it")
        return _graph_register(self, self._input(desc), name)
    if sizes is None:
        raise TypeError("input() needs sizes, or a complete desc=")
    flags = TensorFlags.OWNED_BY_DML if owned else TensorFlags.NONE
    desc = _core.TensorDesc(np.float32 if dtype is None else dtype, sizes, flags=flags,
                            strides=strides)
    return _graph_register(self, self._input(desc), name)


def _graph_constant(self, array, dtype=None, *, sizes=None, name=None):
    """Add an owned input whose data is known now, and return its Expression.

    This is ``input(..., owned=True)`` for the common case, the weight whose
    array is in hand when the graph is built. The graph keeps a reference to
    the array -- not a copy -- until compile(), which uploads it and lets it
    go. A compile() whose every owned input is a constant initializes the
    operator itself, so a graph built from constants alone is ready to
    dispatch as soon as it is compiled.

    Args:
        array: The data: anything np.asarray accepts, or a Buffer.
        dtype: The tensor's element type. Defaults to the array's own dtype.
            The array is converted at upload under the same rules as
            dispatch(); a Buffer's dtype must already match.
        sizes: The tensor's shape. Defaults to the array's; another shape
            with the same element count views the same data through it.
        name: As for input().

    Raises:
        ValueError: The array's dtype would convert unsafely, or does not
            fill the sizes; the name is already taken.
        TypeError: The dtype is not one DirectML supports.
    """
    if not isinstance(array, _core.Buffer):
        array = np.asarray(array)
    target = _NUMPY_DTYPES[_to_data_type(array.dtype if dtype is None else dtype)]
    sizes = list(array.shape) if sizes is None else list(sizes)

    what = f"the constant {name!r}" if name is not None else f"a constant of shape {sizes}"
    if isinstance(array, _core.Buffer):
        if array.dtype != target:
            raise ValueError(f"cannot bind a {array.dtype} Buffer to {what}, which is "
                             f"{target.name}; Buffers are not converted")
    else:
        _check_cast(array, target, what)
    _check_fit(array, target, sizes, 0, False, what)

    desc = _core.TensorDesc(target, sizes, flags=TensorFlags.OWNED_BY_DML)
    expression = _graph_register(self, self._input(desc), name)
    self.__dict__.setdefault("_constants", {})[expression._node_id] = array
    return expression


def _graph_compile(self, outputs, *, flags=ExecutionFlags.NONE):
    """Compile the graph into a CompiledOperator.

    The outputs are fixed here, in this order, and are not named again at
    dispatch. The operator is self-contained: it snapshots what it needs from
    the graph, so the graph can be dropped as soon as this returns.

    The constants the graph recorded go with it. If they are the graph's only
    owned inputs -- or it has none -- the operator is initialized here and
    the graph's references to the arrays are released; otherwise they wait
    for initialize(), which supplies them alongside the other weights.

    Args:
        outputs: The Expressions to compute, in the order dispatch() returns
            them.
        flags: DML_EXECUTION_FLAGS for the compilation.

    Returns:
        A CompiledOperator, ready for dispatch() or, if it has owned inputs
        that are not constants, for initialize().
    """
    op = self._compile(list(outputs), flags=flags)
    op._names = dict(self.__dict__.get("_names", {}))
    op._constants = self.__dict__.pop("_constants", {})

    table = _slot_table(op)
    if all(table.keys[slot.index] in op._constants for slot in table.owned()):
        op.initialize()
    return op


_core.Graph.input = _graph_input
_core.Graph.constant = _graph_constant
_core.Graph.compile = _graph_compile


# --- Elementwise shape checks --------------------------------------------------
# DirectML refuses a mismatched elementwise pair at compile, but only as a bare
# E_INVALIDARG with no node named. The operands are right here, so say which.


def _check_elementwise(symbol, a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"{a!r} {symbol} {b!r}: shapes differ, and nothing broadcasts implicitly; "
            f"use dml.broadcast() to view one through the other's shape")
    if a.dtype != b.dtype:
        raise ValueError(f"{a!r} {symbol} {b!r}: element types differ")


def _checked_operator(name, symbol):
    original = getattr(_core.Expression, name)

    def method(self, other):
        if isinstance(other, _core.Expression):
            _check_elementwise(symbol, self, other)
        return original(self, other)

    method.__name__ = name
    method.__qualname__ = f"Expression.{name}"
    method.__doc__ = original.__doc__
    setattr(_core.Expression, name, method)


for _name, _symbol in [("__add__", "+"), ("__sub__", "-"), ("__mul__", "*"),
                       ("__truediv__", "/"), ("__mod__", "%")]:
    _checked_operator(_name, _symbol)
del _name, _symbol


def add(a, b, *, fused_activation=None):
    """Elementwise ``a + b``, optionally with a fused activation on the result."""
    _check_elementwise("+", a, b)
    return _core.add(a, b, fused_activation=fused_activation)


def subtract(a, b):
    """Elementwise ``a - b``."""
    _check_elementwise("-", a, b)
    return _core.subtract(a, b)


def multiply(a, b):
    """Elementwise ``a * b``."""
    _check_elementwise("*", a, b)
    return _core.multiply(a, b)


def divide(a, b):
    """Elementwise ``a / b``."""
    _check_elementwise("/", a, b)
    return _core.divide(a, b)


# --- Operator wrappers -------------------------------------------------------


def reinterpret(input, sizes, strides=None, dtype=None):
    """View the same bytes through different sizes, strides or dtype.

    ``dtype=None`` keeps the input's type, which is what almost every
    reinterpret wants. The element count implied by the arguments must match
    the input's.
    """
    return _core.reinterpret(input, list(sizes), strides,
                             None if dtype is None else _to_data_type(dtype))


def broadcast(input, shape):
    """View ``input`` as ``shape``, repeating every axis whose extent is 1.

    NumPy's rule, made explicit: axes align from the right, a missing leading
    axis or one of extent 1 repeats to the target's extent, and any other
    difference is an error. The repeat is a stride of 0 -- DirectML reads a
    tensor through whatever strides its descriptor carries -- so nothing is
    copied and no operator is added; the result is a view of the same node.

    Nothing broadcasts implicitly: an elementwise operator wants both operands
    the same shape, and this is how to say which one stretches.

    Args:
        input: The Expression to view.
        shape: The target shape, of the input's rank or higher.

    Raises:
        ValueError: The shapes cannot be broadcast together.
    """
    source = list(input.shape)
    shape = list(shape)
    if source == shape:
        return input
    if len(shape) < len(source):
        raise ValueError(f"cannot broadcast {source} to {shape}: the target has fewer axes")

    # The strides the input is already read through, or packed ones if none.
    strides = input.strides
    if strides is None:
        strides = [1] * len(source)
        for i in range(len(source) - 2, -1, -1):
            strides[i] = strides[i + 1] * source[i + 1]
    strides = list(strides)

    lead = len(shape) - len(source)
    result = [0] * lead
    for size, target, stride in zip(source, shape[lead:], strides):
        if size == target:
            result.append(stride)
        elif size == 1:
            result.append(0)
        else:
            raise ValueError(f"cannot broadcast {source} to {shape}: an axis of {size} "
                             f"cannot repeat to {target}")

    return _core.reinterpret(input, shape, result, None)


def local_response_normalization(input, *, cross_channel, local_size, alpha,
                                 beta, bias):
    """Normalize each element by its neighbourhood's energy, AlexNet-style.

    Computes ``output = input / (bias + (alpha / local_size) * sum) ** beta``
    where ``sum`` adds the squares of the ``local_size`` elements around each
    element.

    Args:
        cross_channel: True takes the window across channels at one pixel;
            False takes it over a square spatial patch within one channel.
        local_size: How many elements the window spans.
        alpha: Scale on the summed squares, divided by ``local_size``.
        beta: The exponent on the whole denominator.
        bias: Constant added to the scaled sum before exponentiation, keeping
            the denominator away from zero.
    """
    return _core.local_response_normalization(
        input, cross_channel=cross_channel, local_size=local_size, alpha=alpha,
        beta=beta, bias=bias)


class MaxPoolingOutputs(typing.NamedTuple):
    values: _core.Expression
    indices: typing.Optional[_core.Expression]


class GRUOutputs(typing.NamedTuple):
    sequence: typing.Optional[_core.Expression]
    single: typing.Optional[_core.Expression]


def max_pooling(input, *, window_sizes, strides=(), start_padding=(),
                end_padding=(), dilations=(), output_indices=False):
    """Max pooling. Returns ``MaxPoolingOutputs(values, indices)``; ``indices``
    is None unless ``output_indices=True``."""
    return MaxPoolingOutputs(*_core.max_pooling(
        input, window_sizes=list(window_sizes), strides=list(strides),
        start_padding=list(start_padding), end_padding=list(end_padding),
        dilations=list(dilations), output_indices=output_indices))


def gru(input, weight, recurrence, bias=None, hidden_init=None,
        sequence_lengths=None, *, activation_descs,
        direction=RecurrentNetworkDirection.FORWARD, linear_before_reset=True,
        output_options=GRUOutputOptions.Both):
    """A one-layer gated recurrent unit. Returns ``GRUOutputs(sequence,
    single)``; each output not requested by ``output_options`` is None."""
    return GRUOutputs(*_core.gru(
        input, weight, recurrence, bias, hidden_init, sequence_lengths,
        activation_descs=list(activation_descs), direction=direction,
        linear_before_reset=linear_before_reset, output_options=output_options))


# --- FusedActivation factories -----------------------------------------------
# One per activation DirectMLX can fuse, so an activation's parameters get
# names instead of riding as bare positional floats -- and an operator that
# cannot be fused never gets constructed as one. Copied 1:1 from DirectMLX.h's
# FusedActivation statics, defaults included.


def _fused_none():
    return FusedActivation(OperatorType.INVALID)

def _fused_elu(alpha=1.0):
    return FusedActivation(OperatorType.ACTIVATION_ELU, alpha)

def _fused_hard_sigmoid(alpha=0.2, beta=0.5):
    return FusedActivation(OperatorType.ACTIVATION_HARD_SIGMOID, alpha, beta)

def _fused_identity():
    return FusedActivation(OperatorType.ACTIVATION_IDENTITY)

def _fused_leaky_relu(alpha=0.01):
    return FusedActivation(OperatorType.ACTIVATION_LEAKY_RELU, alpha)

def _fused_linear(alpha, beta):
    return FusedActivation(OperatorType.ACTIVATION_LINEAR, alpha, beta)

def _fused_parametric_softplus(alpha, beta):
    return FusedActivation(OperatorType.ACTIVATION_PARAMETRIC_SOFTPLUS, alpha, beta)

def _fused_relu():
    return FusedActivation(OperatorType.ACTIVATION_RELU)

def _fused_scaled_elu(alpha=1.67326319217681884765625,
                      gamma=1.05070102214813232421875):
    return FusedActivation(OperatorType.ACTIVATION_SCALED_ELU, alpha, gamma)

def _fused_scaled_tanh(alpha=1.0, beta=0.5):
    return FusedActivation(OperatorType.ACTIVATION_SCALED_TANH, alpha, beta)

def _fused_sigmoid():
    return FusedActivation(OperatorType.ACTIVATION_SIGMOID)

def _fused_softplus(steepness=1.0):
    return FusedActivation(OperatorType.ACTIVATION_SOFTPLUS, steepness)

def _fused_softsign():
    return FusedActivation(OperatorType.ACTIVATION_SOFTSIGN)

def _fused_tanh():
    return FusedActivation(OperatorType.ACTIVATION_TANH)

def _fused_thresholded_relu(alpha=1.0):
    return FusedActivation(OperatorType.ACTIVATION_THRESHOLDED_RELU, alpha)

def _fused_shrink(bias=0.0, threshold=0.5):
    return FusedActivation(OperatorType.ACTIVATION_SHRINK, bias, threshold)

def _fused_celu(alpha=1.0):
    return FusedActivation(OperatorType.ACTIVATION_CELU, alpha)

def _fused_gelu():
    return FusedActivation(OperatorType.ACTIVATION_GELU)


for _name, _factory in [
    ("none", _fused_none), ("elu", _fused_elu),
    ("hard_sigmoid", _fused_hard_sigmoid), ("identity", _fused_identity),
    ("leaky_relu", _fused_leaky_relu), ("linear", _fused_linear),
    ("parametric_softplus", _fused_parametric_softplus), ("relu", _fused_relu),
    ("scaled_elu", _fused_scaled_elu), ("scaled_tanh", _fused_scaled_tanh),
    ("sigmoid", _fused_sigmoid), ("softplus", _fused_softplus),
    ("softsign", _fused_softsign), ("tanh", _fused_tanh),
    ("thresholded_relu", _fused_thresholded_relu), ("shrink", _fused_shrink),
    ("celu", _fused_celu), ("gelu", _fused_gelu),
]:
    _factory.__name__ = _name
    _factory.__qualname__ = f"FusedActivation.{_name}"
    setattr(FusedActivation, _name, staticmethod(_factory))
del _name, _factory
