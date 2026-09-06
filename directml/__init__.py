"""Python binding for DirectML.

``_core`` is the compiled extension and owns resources, execution and the data
hot path; this wrapper layer owns signature shaping, defaults, validation and
error messages. Every class here is the ``_core`` class itself, with the
wrapper's methods and properties attached at import time: instances are created
on the C++ side, so a Python subclass would only cover the objects users
construct and not the ones the library hands back.
"""

import functools
import importlib.metadata
import math
import typing

import numpy as np

from . import _core
from ._core import *  # noqa: F401,F403

try:
    __version__ = importlib.metadata.version("directml")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


# How a DML_TENSOR_DATA_TYPE is spelled in numpy; every API that takes a data
# type accepts either spelling.
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

    A Buffer is a tensor in the device's memory. A binding dict accepts one
    wherever it accepts an array, and binds it in place with no upload;
    ``dispatch(readback=False)`` returns one per output instead of copying the
    outputs back; ``numpy()`` copies it back explicitly.

    Args:
        device: The Device whose memory to use.
        array: The data; anything np.asarray accepts.
        dtype: The buffer's element type. Defaults to the array's own dtype.
            The array is converted under the same rules as dispatch().

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
    """Refuse a conversion that crosses dtype kinds unless NumPy calls it safe.

    Narrowing within a kind (float64 to float32, float32 to float16) is
    allowed. NumPy's own 'same_kind' rule would also allow int32 to float32,
    which is the silent-wrong-answer case this exists to catch.
    """
    if (array.dtype != target and array.dtype.kind != target.kind
            and not np.can_cast(array.dtype, target, "safe")):
        raise ValueError(
            f"cannot convert a {array.dtype} array to {target.name} for {what}; "
            f"convert it explicitly with astype()")


def _check_count(array, sizes, what):
    """Refuse an array or Buffer whose element count differs from ``sizes``."""
    if array.size != math.prod(sizes):
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

    def check_fit(self, value):
        """Refuse data that does not fill this input's tensor."""
        if not self.strided:
            _check_count(value, self.sizes, self.describe())
            return
        # A strided view repeats or reorders elements, so the data covers the
        # underlying buffer rather than the logical shape.
        nbytes = value.nbytes if isinstance(value, _core.Buffer) else value.size * self.dtype.itemsize
        if nbytes < self.total_bytes:
            raise ValueError(
                f"{nbytes} bytes do not fill {self.describe()}, whose buffer holds "
                f"{self.total_bytes} bytes")


class _Slots:
    """A compiled operator's inputs, by index, node identity and name."""

    def __init__(self, op):
        slots = op._input_slots
        self.keys = [key for key, _, _ in slots]
        self.by_index = [
            _Slot(index, owned, desc, op._names.get(key))
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
    """Match a binding dict against the graph's inputs.

    Returns ``(buffers, staged)``: the inputs supplied as Buffers and the
    arrays still to convert and upload, both by input index. Every error is
    raised from here, before any upload starts. ``defaults`` supplies the
    constants the graph recorded for any owned input the dict leaves out.
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
            if value.dtype != slot.dtype:
                raise ValueError(
                    f"cannot bind a {value.dtype} Buffer to {slot.describe()}; "
                    f"Buffers are not converted")
            slot.check_fit(value)
            buffers[slot.index] = value
            return

        array = np.asarray(value)
        _check_cast(array, slot.dtype, slot.describe())
        slot.check_fit(array)
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

    Conversion happens one tensor at a time as the upload loop in _core pulls
    on this generator, so only one converted copy is alive at once.
    """
    slots = _slot_table(op).by_index
    for index, array in staged.items():
        yield index, np.ascontiguousarray(array, slots[index].dtype)


# --- CompiledOperator: dict-shaped initialize and dispatch -------------------


def _operator_initialize(self, weights=None):
    """Upload the owned inputs and run the operator initializer.

    The data is read once, here, and lives on the GPU from then on; the
    library keeps no copy. To change an owned input's data afterwards, call
    initialize() again with every owned input, constants included.

    Args:
        weights: A dict mapping owned inputs to their data, keyed by
            Expression or by the name given at ``graph.input()``. A value is
            an array, converted to the tensor's dtype under the same rules as
            dispatch(), or a Buffer, bound as it is. Inputs declared with
            ``graph.constant()`` may be left out the first time.

    Raises:
        ValueError: A key is not an input of this graph, is not owned, or is
            bound twice; an owned input is missing; an array's dtype would
            convert unsafely; a Buffer's dtype differs from the tensor's; or
            the data does not fit.
    """
    buffers, staged = _validate(self, weights or {}, owned=True, verb="initialize",
                                defaults=self._constants)
    self._initialize(buffers, _converted(self, staged))
    self._constants = {}


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
            GPU and returns a Buffer for it.

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


_core_graph_init = _core.Graph.__init__


def _graph_init(self, device, *, tensor_policy=None):
    """A graph under construction.

    Args:
        device: The Device that will compile and run it.
        tensor_policy: A TensorPolicy deciding the layout of the tensors the
            graph creates internally; InterleavedChannel is only reachable
            through it.
    """
    _core_graph_init(self, device, tensor_policy=tensor_policy)
    self._names = {}      # input node id -> the name given at input()/constant()
    self._constants = {}  # input node id -> the array given at constant()


def _graph_register(self, expression, name):
    """Record an input's name on the graph, refusing a duplicate."""
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(f"an input's name must be a str, not {type(name).__name__}")
        if name in self._names.values():
            raise ValueError(f"this graph already has an input named {name!r}")
        self._names[expression._node_id] = name
    return expression


def _graph_input(self, sizes=None, dtype=None, *, owned=False, strides=None, desc=None,
                 name=None):
    """Add an input tensor to the graph and return its Expression.

    Args:
        sizes: The tensor's shape.
        dtype: A numpy dtype or TensorDataType. Defaults to float32.
        owned: Mark the tensor DML_TENSOR_FLAG_OWNED_BY_DML: its data is
            handed over once, at initialize(), and lives on the GPU from
            then on. Without it the data rides along with every dispatch.
            For a weight whose value is already in hand, ``constant()``
            declares an owned input and records the data in one call.
        strides: Element strides for a non-packed view; a stride of 0 repeats
            an axis.
        desc: A complete TensorDesc, for the controls input() does not take
            (total_tensor_size_in_bytes, guaranteed_base_offset_alignment).
            No other tensor argument may be combined with it.
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

    The graph keeps a reference to the array (not a copy) until compile(),
    which uploads it and lets it go. A compile() whose every owned input is a
    constant initializes the operator itself.

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
    _check_count(array, sizes, what)

    desc = _core.TensorDesc(target, sizes, flags=TensorFlags.OWNED_BY_DML)
    expression = _graph_register(self, self._input(desc), name)
    self._constants[expression._node_id] = array
    return expression


def _graph_compile(self, outputs, *, flags=ExecutionFlags.NONE):
    """Compile the graph into a CompiledOperator.

    The outputs are fixed here, in this order. The operator snapshots what it
    needs from the graph, so the graph can be dropped as soon as this returns.
    The constants the graph recorded go with it: if they are the graph's only
    owned inputs, or it has none, the operator is initialized here; otherwise
    they wait for initialize(), which supplies them alongside the other
    weights. Either way the graph no longer holds them.

    Args:
        outputs: The Expressions to compute, in the order dispatch() returns
            them.
        flags: DML_EXECUTION_FLAGS for the compilation.

    Returns:
        A CompiledOperator, ready for dispatch() or, if it has owned inputs
        that are not constants, for initialize().
    """
    op = self._compile(list(outputs), flags=flags)
    op._names = dict(self._names)
    op._constants, self._constants = self._constants, {}

    table = _slot_table(op)
    if all(table.keys[slot.index] in op._constants for slot in table.owned()):
        op.initialize()
    return op


_core.Graph.__init__ = _graph_init
_core.Graph.input = _graph_input
_core.Graph.constant = _graph_constant
_core.Graph.compile = _graph_compile


# --- Elementwise operators, checked where they are written ------------------
# DirectML refuses a mismatched elementwise pair at compile, but only as a bare
# E_INVALIDARG with no node named. The operands are right here, so say which.


def _check_elementwise(where, a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"{where}: shapes differ, and nothing broadcasts implicitly; "
            f"use dml.broadcast() to view one through the other's shape")
    if a.dtype != b.dtype:
        raise ValueError(f"{where}: element types differ")


def _checked(symbol, function):
    """``function`` with the shape and dtype check in front of it."""
    @functools.wraps(function)
    def checked(a, b, **kwargs):
        _check_elementwise(f"{a!r} {symbol} {b!r}", a, b)
        return function(a, b, **kwargs)
    return checked


def _checked_call(function):
    """The same check for an operator written as a call, not a symbol."""
    name = function.__name__

    @functools.wraps(function)
    def checked(a, b, **kwargs):
        _check_elementwise(f"{name}({a!r}, {b!r})", a, b)
        return function(a, b, **kwargs)
    return checked


def _with_dtype(function, *names):
    """``function`` with the named data type keywords in either spelling."""
    @functools.wraps(function)
    def converting(*args, **kwargs):
        for name in names:
            if kwargs.get(name) is not None:
                kwargs[name] = _to_data_type(kwargs[name])
        return function(*args, **kwargs)
    return converting


add = _checked("+", _core.add)
subtract = _checked("-", _core.subtract)
multiply = _checked("*", _core.multiply)
divide = _checked("/", _core.divide)

# The elementwise operators with no symbol of their own. Comparisons write
# their result as output_dtype, which is uint8 unless asked otherwise.
max = _checked_call(_core.max)
min = _checked_call(_core.min)
mean = _checked_call(_core.mean)
atan_yx = _checked_call(_core.atan_yx)
difference_square = _checked_call(_core.difference_square)
logical_and = _checked_call(_core.logical_and)
logical_or = _checked_call(_core.logical_or)
logical_xor = _checked_call(_core.logical_xor)
bit_and = _checked_call(_core.bit_and)
bit_or = _checked_call(_core.bit_or)
bit_xor = _checked_call(_core.bit_xor)
bit_shift_left = _checked_call(_core.bit_shift_left)
bit_shift_right = _checked_call(_core.bit_shift_right)
modulus_truncate = _checked_call(_core.modulus_truncate)
modulus_floor = _checked_call(_core.modulus_floor)
equals = _checked_call(_with_dtype(_core.equals, "output_dtype"))
greater_than = _checked_call(_with_dtype(_core.greater_than, "output_dtype"))
greater_than_or_equal = _checked_call(_with_dtype(_core.greater_than_or_equal, "output_dtype"))
less_than = _checked_call(_with_dtype(_core.less_than, "output_dtype"))
less_than_or_equal = _checked_call(_with_dtype(_core.less_than_or_equal, "output_dtype"))

# The unary operators that name a data type, in either spelling.
is_nan = _with_dtype(_core.is_nan, "output_dtype")
is_infinity = _with_dtype(_core.is_infinity, "output_dtype")
bit_count = _with_dtype(_core.bit_count, "output_dtype")
quantize_linear = _with_dtype(_core.quantize_linear, "output_dtype")


def _patch_operator(name, function):
    """Route ``Expression <op> Expression`` through the checked ``function``.

    A float operand keeps DirectMLX's overload, which has nothing to check.
    """
    original = getattr(_core.Expression, name)

    def method(self, other):
        if isinstance(other, _core.Expression):
            return function(self, other)
        return original(self, other)

    method.__name__ = name
    method.__qualname__ = f"Expression.{name}"
    method.__doc__ = original.__doc__
    setattr(_core.Expression, name, method)


for _name, _function in [
    ("__add__", add), ("__sub__", subtract), ("__mul__", multiply),
    ("__truediv__", divide), ("__mod__", _checked("%", _core.Expression.__mod__)),
]:
    _patch_operator(_name, _function)
del _name, _function


# --- Operator wrappers -------------------------------------------------------


def reinterpret(input, sizes, strides=None, dtype=None):
    """View the same bytes through different sizes, strides or dtype.

    ``dtype=None`` keeps the input's type. The element count implied by the
    arguments must match the input's.
    """
    return _core.reinterpret(input, list(sizes), strides,
                             None if dtype is None else _to_data_type(dtype))


def broadcast(input, shape):
    """View ``input`` as ``shape``, repeating every axis whose extent is 1.

    NumPy's rule: axes align from the right, a missing leading axis or one of
    extent 1 repeats to the target's extent, and any other difference is an
    error. The repeat is a stride of 0, so nothing is copied and no operator
    is added; the result is a view of the same node. Nothing broadcasts
    implicitly: an elementwise operator wants both operands the same shape,
    and this is how to say which one stretches.

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


def pow(input, exponent, *, scale_bias=None):
    """``input ** exponent``, elementwise.

    The exponent is a tensor of the input's shape and type, or a Python float,
    which is a different DirectML operator and not a constant tensor.
    """
    if isinstance(exponent, Expression):
        _check_elementwise(f"pow({input!r}, {exponent!r})", input, exponent)
    return _core.pow(input, exponent, scale_bias=scale_bias)


def where(condition, a, b):
    """``a`` where ``condition`` is non-zero, ``b`` elsewhere, elementwise.

    DirectMLX spells this ``If``. All three tensors want the same shape, and
    the condition is a uint8 tensor whose zero elements are the false ones.
    """
    _check_elementwise(f"where({condition!r}, {a!r}, {b!r})", a, b)
    if condition.shape != a.shape:
        raise ValueError(
            f"where({condition!r}, {a!r}, {b!r}): the condition's shape differs "
            f"from the operands'; use dml.broadcast() to view it through theirs")
    return _core.where(condition, a, b)


def cast(input, *, dtype):
    """The input's values converted to ``dtype``, as a ``static_cast`` would.

    This is the operator that rounds and truncates; ``reinterpret`` is the one
    that rereads the same bytes.
    """
    return _core.cast(input, dtype=_to_data_type(dtype))


def reduce(input, *, function, axes=(), output_dtype=None):
    """Reduce ``axes`` to an extent of 1 with ``function``.

    Args:
        function: A ReduceFunction: SUM, AVERAGE, MAX, ARGMAX, L2, ...
        axes: The axes to reduce; every axis by default.
        output_dtype: The result's type, the input's by default. ARGMAX and
            ARGMIN write indices and want an integer type here.
    """
    return _core.reduce(
        input, function=function, axes=list(axes),
        output_dtype=(TensorDataType.UNKNOWN if output_dtype is None
                      else _to_data_type(output_dtype)))


def one_hot(indices, values, *, output_length, axis):
    """A tensor of ``values[1]`` at each index and ``values[0]`` elsewhere.

    Args:
        indices: The index to set, per position of the other axes.
        values: A two-element tensor, [off, on]. Its type is the result's.
        output_length: The extent the active axis grows to.
        axis: The axis the indices point into.
    """
    return _core.one_hot(indices, values, output_length=output_length, axis=axis)


def gather_nd(input, indices, *, input_dimension_count, indices_dimension_count,
              batch_dimension_count=0):
    """Slices of the input, addressed by coordinates in the indices tensor.

    Args:
        input_dimension_count: How many trailing axes of the input the
            coordinates address; the rest are batch axes.
        indices_dimension_count: How many trailing axes of the indices tensor
            hold coordinates, the last of them being the coordinate itself.
        batch_dimension_count: How many leading axes the two tensors share,
            gathered one batch at a time.
    """
    return _core.gather_nd(
        input, indices, input_dimension_count=input_dimension_count,
        indices_dimension_count=indices_dimension_count,
        batch_dimension_count=batch_dimension_count)


def scatter_nd(input, indices, updates, *, input_dimension_count,
               indices_dimension_count):
    """A copy of the input with ``updates`` written at ``indices``.

    The dimension counts are ``gather_nd``'s, and the updates tensor holds one
    slice per coordinate.
    """
    return _core.scatter_nd(
        input, indices, updates, input_dimension_count=input_dimension_count,
        indices_dimension_count=indices_dimension_count)


def resample(input, *, output_sizes, mode,
             rounding_direction=AxisDirection.INCREASING, scales=(),
             input_pixel_offsets=(), output_pixel_offsets=(), antialiased=False):
    """Resample every axis of the input to ``output_sizes``.

    ``upsample_2d`` is the same operator with an integer scale on the last two
    axes; this one takes the shape it should produce.

    Args:
        output_sizes: The shape out, of the input's rank.
        mode: NEAREST_NEIGHBOR or LINEAR.
        rounding_direction: Which way NEAREST_NEIGHBOR breaks a tie; INCREASING
            by default.
        scales: Output extent over input extent, per axis. Computed from the
            sizes when empty.
        input_pixel_offsets: Where a source pixel's center sits, per axis;
            0.5 each when empty.
        output_pixel_offsets: The same for the destination; -0.5 each when
            empty. Together with the scales these decide the sampling grid.
        antialiased: Filter over the whole source window when downsampling,
            rather than point-sampling it.
    """
    return _core.resample(
        input, output_sizes=list(output_sizes), mode=mode,
        rounding_direction=rounding_direction, scales=list(scales),
        input_pixel_offsets=list(input_pixel_offsets),
        output_pixel_offsets=list(output_pixel_offsets), antialiased=antialiased)


def roi_align(input, roi, batch_indices, *, reduction_function,
              interpolation_mode, spatial_scale_x, spatial_scale_y,
              input_pixel_offset, output_pixel_offset, out_of_bounds_input_value,
              minimum_samples_per_output, maximum_samples_per_output,
              align_regions_to_corners, output_height, output_width):
    """Pool each region of interest down to one fixed-size tile.

    Args:
        input: The feature map, [batch, channels, height, width].
        roi: One box per region, as x1, y1, x2, y2 along the last axis.
        batch_indices: Which image of the batch each region belongs to.
        reduction_function: AVERAGE or MAX over the samples of a bin.
        interpolation_mode: How a sample between pixels is read.
        spatial_scale_x: Scale from the boxes' coordinates to input pixels,
            horizontally.
        spatial_scale_y: The same, vertically.
        input_pixel_offset: Where a pixel's center sits in the input, usually
            0.5.
        output_pixel_offset: The same for an output bin, usually -0.5.
        out_of_bounds_input_value: What a sample outside the input reads as.
        minimum_samples_per_output: Fewest samples taken per output bin.
        maximum_samples_per_output: Most samples taken per output bin; equal to
            the minimum fixes the sampling ratio.
        align_regions_to_corners: Align the regions to pixel corners rather
            than centers, dropping the half-pixel shift.
        output_height: Rows of the output tile.
        output_width: Columns of the output tile.
    """
    return _core.roi_align(
        input, roi, batch_indices, reduction_function=reduction_function,
        interpolation_mode=interpolation_mode, spatial_scale_x=spatial_scale_x,
        spatial_scale_y=spatial_scale_y, input_pixel_offset=input_pixel_offset,
        output_pixel_offset=output_pixel_offset,
        out_of_bounds_input_value=out_of_bounds_input_value,
        minimum_samples_per_output=minimum_samples_per_output,
        maximum_samples_per_output=maximum_samples_per_output,
        align_regions_to_corners=align_regions_to_corners,
        output_height=output_height, output_width=output_width)


def _fill_value(value, data_type, what):
    """``value`` as the eight bytes DirectML reads a scalar of ``data_type`` from."""
    try:
        scalar = np.array(value, dtype=_NUMPY_DTYPES[data_type])
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{what}={value!r} is not a {_NUMPY_DTYPES[data_type]}: "
                         f"{error}") from None
    return scalar.tobytes().ljust(8, b"\0")


def fill_value_constant(graph, *, sizes, value, dtype=None):
    """A tensor of ``sizes`` filled with ``value``, computed on the GPU.

    Nothing is uploaded and no input is added: this is the constant to reach
    for when a graph needs a tensor of zeros or ones, rather than
    ``graph.constant(np.zeros(...))``.

    Args:
        sizes: The shape to produce.
        value: The value every element takes, converted to ``dtype``.
        dtype: The tensor's type; float32 by default.
    """
    data_type = TensorDataType.FLOAT32 if dtype is None else _to_data_type(dtype)
    return _core.fill_value_constant(
        graph, sizes=list(sizes), dtype=data_type,
        value=_fill_value(value, data_type, "value"))


def fill_value_sequence(graph, *, sizes, value_start, value_delta, dtype=None):
    """A tensor of ``sizes`` counting from ``value_start`` by ``value_delta``.

    The sequence runs in memory order, so a [1, 1, 2, 3] tensor starting at 0
    with a delta of 1 holds [[0, 1, 2], [3, 4, 5]].

    DirectML's graph compiler faults on a graph whose output is this operator
    itself; pass the result through another operator (``dml.identity()`` will
    do) when the sequence is what the graph returns.

    Args:
        sizes: The shape to produce.
        value_start: The first element's value.
        value_delta: What each element adds to the one before it.
        dtype: The tensor's type; float32 by default.
    """
    data_type = TensorDataType.FLOAT32 if dtype is None else _to_data_type(dtype)
    return _core.fill_value_sequence(
        graph, sizes=list(sizes), dtype=data_type,
        value_start=_fill_value(value_start, data_type, "value_start"),
        value_delta=_fill_value(value_delta, data_type, "value_delta"))


class TopKOutputs(typing.NamedTuple):
    values: _core.Expression
    indices: _core.Expression


class NonZeroCoordinatesOutputs(typing.NamedTuple):
    count: _core.Expression
    coordinates: _core.Expression


class RandomGeneratorOutputs(typing.NamedTuple):
    values: _core.Expression
    state: typing.Optional[_core.Expression]


def top_k(input, *, axis, k, axis_direction=AxisDirection.DECREASING):
    """The ``k`` largest elements along ``axis``, and where they came from.

    Returns ``TopKOutputs(values, indices)``, both of the input's shape with
    ``axis`` cut to ``k``. ``axis_direction=AxisDirection.INCREASING`` takes the
    smallest instead. DirectMLX names these outputs value and index; they are
    plural here, as max_pooling's are.
    """
    return TopKOutputs(*_core.top_k(input, axis=axis, k=k,
                                    axis_direction=axis_direction))


def non_zero_coordinates(input):
    """Where the input's non-zero elements are.

    Returns ``NonZeroCoordinatesOutputs(count, coordinates)``: how many
    elements are non-zero, and a [element count, rank] tensor of their indices,
    of which only the first ``count`` rows are filled in.
    """
    return NonZeroCoordinatesOutputs(*_core.non_zero_coordinates(input))


def random_generator(input_state, *, output_sizes, output_state=True,
                     type=RandomGeneratorType.PHILOX_4X32_10):
    """Uniform random uint32s, from a generator state tensor.

    Returns ``RandomGeneratorOutputs(values, state)``; ``state`` is None unless
    ``output_state=True``, and is the state to feed the next dispatch, which is
    what makes the stream continue rather than repeat.
    """
    return RandomGeneratorOutputs(*_core.random_generator(
        input_state, output_sizes=list(output_sizes), output_state=output_state,
        type=type))


def dequantize(input, quantization_parameters, *, quantization_type):
    """Dequantize the input, block by block.

    The general form of ``dequantize_linear``: the parameters may cover fewer
    elements than the input, in which case each one applies to a whole block of
    it, which is how a 4-bit weight matrix carries a scale per group of
    elements rather than per element.

    Args:
        quantization_parameters: ``[scale]`` for
            ``QuantizationType.SCALE``, ``[scale, zero_point]`` for
            ``QuantizationType.SCALE_ZERO_POINT``. The output takes the
            scale's data type.
        quantization_type: Which of those two forms the list is in. Without a
            zero point an unsigned input is read as having an implicit one at
            the middle of its range, so uint8 0 comes back as ``-128 * scale``;
            a signed input is taken at face value.
    """
    return _core.dequantize(input, list(quantization_parameters),
                            quantization_type=quantization_type)


def convolution_integer(input, filter, input_zero_point=None,
                        filter_zero_point=None, *, strides=(), dilations=(),
                        start_padding=(), end_padding=(), group_count=1,
                        output_sizes=()):
    """Convolve an integer input with an integer filter, summing into int32.

    The zero points are subtracted before the multiply, so the operator is
    ``convolution(input - input_zero_point, filter - filter_zero_point)`` in
    exact integer arithmetic. Nothing is requantized: the output is the int32
    accumulator, which the caller scales itself.

    Args:
        input_zero_point: What value of the input stands for zero; a scalar
            tensor, or one per input channel. None means zero.
        filter_zero_point: The same for the filter. None means zero.

    The remaining arguments are ``convolution``'s, minus the modes it has no
    use for here.
    """
    return _core.convolution_integer(
        input, filter, input_zero_point, filter_zero_point,
        strides=list(strides), dilations=list(dilations),
        start_padding=list(start_padding), end_padding=list(end_padding),
        group_count=group_count, output_sizes=list(output_sizes))


def quantized_linear_convolution(input, input_scale, filter, filter_scale,
                                 output_scale, input_zero_point=None,
                                 filter_zero_point=None, bias=None,
                                 output_zero_point=None, *, output_dtype,
                                 strides=(), dilations=(), start_padding=(),
                                 end_padding=(), group_count=1,
                                 output_sizes=()):
    """Convolve in the quantized domain and requantize the result.

    ONNX's QLinearConv: the convolution runs on the integers as
    ``convolution_integer`` does, the ``bias`` is added to that integer
    accumulator, and the sum is scaled by
    ``input_scale * filter_scale / output_scale``, rounded, offset by
    ``output_zero_point`` and saturated to ``output_dtype``.

    Args:
        input_scale: What one unit of the input is worth; a scalar tensor, or
            one per input channel.
        filter_scale: The same for the filter, usually one per output channel.
        output_scale: The same for the output.
        input_zero_point: What value stands for zero, alongside each scale.
            None means zero.
        filter_zero_point: The same for the filter. None means zero.
        bias: An int32 bias per output channel, in units of
            ``input_scale * filter_scale``, not of the output.
        output_zero_point: The same for the output; its data type must be
            ``output_dtype``.
        output_dtype: int8 or uint8, the type the result saturates to.

    The remaining arguments are ``convolution``'s.
    """
    return _core.quantized_linear_convolution(
        input, input_scale, filter, filter_scale, output_scale,
        input_zero_point, filter_zero_point, bias, output_zero_point,
        output_dtype=_to_data_type(output_dtype), strides=list(strides),
        dilations=list(dilations), start_padding=list(start_padding),
        end_padding=list(end_padding), group_count=group_count,
        output_sizes=list(output_sizes))


# --- Gradients ---------------------------------------------------------------
# DirectML implements the backward pass of a handful of operators. Nothing in
# this library differentiates a graph; each of these is one link, and the chain
# rule belongs to whoever is writing the training step.


class BatchNormalizationGradOutputs(typing.NamedTuple):
    gradient: _core.Expression
    scale_gradient: _core.Expression
    bias_gradient: _core.Expression


class BatchNormalizationTrainingOutputs(typing.NamedTuple):
    output: _core.Expression
    mean: _core.Expression
    variance: _core.Expression


class RoiAlignGradOutputs(typing.NamedTuple):
    gradient: typing.Optional[_core.Expression]
    roi_gradient: typing.Optional[_core.Expression]


def batch_normalization_training(input, scale, bias, fused_add=None, *,
                                 epsilon=1e-5, fused_activation=None):
    """Batch normalization over statistics taken from the batch itself.

    ``batch_normalization`` is handed the mean and variance to normalize by;
    this one computes them, per channel, over every other axis, and returns
    them so the backward pass and the running averages can have them.

    Returns ``BatchNormalizationTrainingOutputs(output, mean, variance)``. The
    variance is the biased one, divided by the element count rather than by one
    less than it.

    Args:
        fused_add: A tensor added to the result, after the scale and the bias,
            which is how a residual connection folds into this operator. It
            does not reach the statistics: those are the input's own. None to
            add nothing.
    """
    return BatchNormalizationTrainingOutputs(*_core.batch_normalization_training(
        input, scale, bias, fused_add, epsilon=epsilon,
        fused_activation=fused_activation))


def batch_normalization_grad(input, input_gradient, mean, variance, scale, *,
                             epsilon=1e-5):
    """The gradient of ``batch_normalization``.

    The mean and variance are constants here, as they are in inference: the
    gradient towards the input is just ``input_gradient * scale /
    sqrt(variance + epsilon)``. ``batch_normalization_training_grad`` is the
    one that accounts for the statistics being functions of the input.

    Returns ``BatchNormalizationGradOutputs(gradient, scale_gradient,
    bias_gradient)``, of the input's shape and of the mean's shape twice.
    """
    return BatchNormalizationGradOutputs(*_core.batch_normalization_grad(
        input, input_gradient, mean, variance, scale, epsilon=epsilon))


def batch_normalization_training_grad(input, input_gradient, mean, variance,
                                      scale, *, epsilon=1e-5):
    """The gradient of ``batch_normalization_training``.

    Pass the mean and variance that operator returned. They came from the
    input, so the gradient towards it carries the two extra terms that
    ``batch_normalization_grad`` leaves out.

    Returns ``BatchNormalizationGradOutputs(gradient, scale_gradient,
    bias_gradient)``.
    """
    return BatchNormalizationGradOutputs(*_core.batch_normalization_training_grad(
        input, input_gradient, mean, variance, scale, epsilon=epsilon))


def resample_grad(input_gradient, *, output_sizes, mode, scales=(),
                  input_pixel_offsets=(), output_pixel_offsets=()):
    """The gradient of ``resample``.

    Every output element of the forward pass contributed to one input element
    (NEAREST_NEIGHBOR) or to a few (LINEAR); this sums the gradients back onto
    the elements they were read from, so ``output_sizes`` is the shape the
    forward pass consumed.

    The sampling grid must be described the same way it was on the way
    forward, except that the defaults differ: here they are 0.5 and -0.5, the
    half-pixel grid, and ``resample``'s own default computes the scales from
    the sizes instead. Pass all three explicitly when the forward call did.

    Args:
        output_sizes: The shape of the tensor the forward pass took in.
        mode: NEAREST_NEIGHBOR or LINEAR, as on the way forward.
        scales: Output extent over input extent, per axis, of the forward
            pass. Computed from the sizes when empty.
        input_pixel_offsets: 0.5 each when empty.
        output_pixel_offsets: -0.5 each when empty.
    """
    return _core.resample_grad(
        input_gradient, output_sizes=list(output_sizes), mode=mode,
        scales=list(scales), input_pixel_offsets=list(input_pixel_offsets),
        output_pixel_offsets=list(output_pixel_offsets))


def roi_align_grad(input_gradient, roi, batch_indices, input=None, *,
                   reduction_function, interpolation_mode, spatial_scale_x,
                   spatial_scale_y, input_pixel_offset, output_pixel_offset,
                   minimum_samples_per_output, maximum_samples_per_output,
                   align_regions_to_corners, batch_size, image_height,
                   image_width, compute_output_gradient=True,
                   compute_output_roi_gradient=False):
    """The gradient of ``roi_align``, towards the feature map and the boxes.

    Returns ``RoiAlignGradOutputs(gradient, roi_gradient)``; each is None
    unless its ``compute_`` flag asked for it, and at least one flag must be
    set. ``gradient`` is [batch_size, channels, image_height, image_width] —
    the forward pass consumed a feature map, not a shape this operator can
    infer, so its shape is spelled out here.

    Args:
        input_gradient: The gradient arriving at ``roi_align``'s output, one
            tile per region.
        roi: The boxes the forward pass pooled, unchanged.
        batch_indices: Which image each region belongs to, unchanged.
        input: The feature map the forward pass read. Required by
            ReduceFunction.MAX, which has to find which element won, and by
            ``compute_output_roi_gradient``, since a box moves along the slope
            of the map. None is only for an AVERAGE pooling's feature-map
            gradient.
        batch_size: Images in the feature map.
        image_height: Rows of the feature map.
        image_width: Columns of the feature map.
        compute_output_gradient: Produce the gradient towards the feature map.
        compute_output_roi_gradient: Produce the gradient towards the boxes.

    The rest are ``roi_align``'s own parameters and must match the values it
    was given.
    """
    return RoiAlignGradOutputs(*_core.roi_align_grad(
        input_gradient, roi, batch_indices, input,
        reduction_function=reduction_function,
        interpolation_mode=interpolation_mode,
        spatial_scale_x=spatial_scale_x, spatial_scale_y=spatial_scale_y,
        input_pixel_offset=input_pixel_offset,
        output_pixel_offset=output_pixel_offset,
        minimum_samples_per_output=minimum_samples_per_output,
        maximum_samples_per_output=maximum_samples_per_output,
        align_regions_to_corners=align_regions_to_corners,
        batch_size=batch_size, image_height=image_height,
        image_width=image_width,
        compute_output_gradient=compute_output_gradient,
        compute_output_roi_gradient=compute_output_roi_gradient))

# --- FusedActivation factories -----------------------------------------------
# One per activation DirectMLX can fuse, copied 1:1 from DirectMLX.h's
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
