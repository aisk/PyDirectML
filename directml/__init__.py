"""Python binding for DirectML.

The package splits in two along the line drawn in docs/api-design.md: ``_core``
is the compiled extension and owns resources, execution and the data hot path;
this wrapper layer owns signature shaping, defaults, validation and error
messages. The boundary of the library is ``import directml``, not the ``.pyd``.
"""

import importlib.metadata
import math

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


# --- Expression conveniences -------------------------------------------------


def _expression_shape(self):
    return tuple(self.desc.sizes)


def _expression_strides(self):
    strides = self.desc.strides
    return None if strides is None else tuple(strides)


def _expression_dtype(self):
    return _NUMPY_DTYPES[self.desc.data_type]


def _expression_repr(self):
    desc = self.desc
    return f"<dml.Expression {_NUMPY_DTYPES[desc.data_type].name} {list(desc.sizes)}>"


_core.Expression.shape = property(
    _expression_shape, doc="The shape of this expression's output, as a tuple.")
_core.Expression.strides = property(
    _expression_strides,
    doc="The element strides of this expression's output, or None if packed.")
_core.Expression.dtype = property(
    _expression_dtype, doc="The element type of this expression's output, as a numpy dtype.")
_core.Expression.__repr__ = _expression_repr


# --- CompiledOperator: dict-shaped initialize and dispatch -------------------


class _Slot:
    """What the wrapper needs to know about one graph input."""

    __slots__ = ("index", "owned", "dtype", "sizes", "total_bytes", "strided")

    def __init__(self, index, owned, desc):
        self.index = index
        self.owned = owned
        self.dtype = _NUMPY_DTYPES[desc.data_type]
        self.sizes = tuple(desc.sizes)
        self.total_bytes = desc.total_tensor_size_in_bytes
        self.strided = desc.strides is not None

    def describe(self):
        return f"input {self.index} ({self.dtype.name} {list(self.sizes)})"


def _slot_table(op):
    table = getattr(op, "_slot_cache", None)
    if table is None:
        table = {
            key: _Slot(index, owned, desc)
            for index, (key, owned, desc) in enumerate(op._input_slots)
        }
        op._slot_cache = table
    return table


def _validate(op, inputs, owned, verb):
    """Match the dict against the graph's inputs and return {index: array}.

    Everything that can go wrong is raised from here, before any upload starts:
    an expression that is not an input, one bound in the wrong phase, a dtype
    that would silently lose meaning, an array of the wrong size, and inputs
    that are missing outright.
    """
    table = _slot_table(op)
    staged = {}

    for expression, array in inputs.items():
        if not isinstance(expression, _core.Expression):
            raise TypeError(
                f"{verb}() keys must be Expressions, not {type(expression).__name__}")

        slot = table.get(expression._node_id)
        if slot is None:
            raise ValueError(f"{expression!r} is not an input of this graph")
        if slot.owned != owned:
            if owned:
                raise ValueError(
                    f"{expression!r} is not owned by DirectML; bind it at dispatch instead")
            raise ValueError(
                f"{expression!r} is owned by DirectML; bind it in initialize() instead")

        array = np.asarray(array)
        target = slot.dtype

        # Narrowing within a kind is fine -- float64 to float32 is what
        # np.zeros() lands on, float32 to float16 is how half-precision weights
        # get loaded -- and so is any cast NumPy calls safe. Crossing kinds any
        # other way, int32 into a float32 tensor above all, is the
        # silent-wrong-answer case, and has to be spelled out with an explicit
        # astype. NumPy's own 'same_kind' rule is no help here: it permits
        # int32 to float32.
        if (array.dtype != target and array.dtype.kind != target.kind
                and not np.can_cast(array.dtype, target, "safe")):
            raise ValueError(
                f"cannot bind a {array.dtype} array to a {target.name} tensor "
                f"({expression!r}); convert it explicitly with astype()")

        if slot.strided:
            # A strided view repeats or reorders elements, so the array covers
            # the underlying buffer rather than the logical shape.
            if array.size * target.itemsize > slot.total_bytes:
                raise ValueError(
                    f"array of {array.size * target.itemsize} bytes does not fit "
                    f"{expression!r}, whose buffer holds {slot.total_bytes} bytes")
        elif array.size != math.prod(slot.sizes):
            raise ValueError(
                f"array of {array.size} elements does not fill {expression!r}")

        staged[slot.index] = array

    missing = [
        slot for slot in table.values()
        if slot.owned == owned and slot.index not in staged
    ]
    if missing:
        raise ValueError(
            f"{verb}() is missing " + ", ".join(slot.describe() for slot in sorted(
                missing, key=lambda slot: slot.index)))

    return staged


def _converted(op, staged):
    """Yield (index, contiguous array of the tensor's dtype) pairs.

    Conversion happens one tensor at a time, as the upload loop in _core pulls
    on this generator: converting the whole dict first would hold a second copy
    of every weight at once, which for model-sized graphs is gigabytes.
    """
    slots = {slot.index: slot for slot in _slot_table(op).values()}
    for index, array in staged.items():
        yield index, np.ascontiguousarray(array, slots[index].dtype)


def _operator_initialize(self, weights):
    """Upload the OWNED_BY_DML inputs and run the operator initializer.

    ``weights`` maps each owned input Expression to its array. The data is read
    once, here, and lives on the GPU from then on; the library keeps no copy.
    To change an owned input's data afterwards, call initialize() again.
    """
    staged = _validate(self, weights, owned=True, verb="initialize")
    self._initialize(_converted(self, staged))


def _operator_dispatch(self, inputs=None):
    """Run the operator and return its outputs as numpy arrays.

    ``inputs`` maps each non-owned input Expression to its array. A graph
    without owned inputs initializes itself on the first call; one with owned
    inputs must be initialize()d explicitly first.
    """
    table = _slot_table(self)
    if not self.initialized:
        owned = sorted((slot for slot in table.values() if slot.owned),
                       key=lambda slot: slot.index)
        if owned:
            raise ValueError(
                "initialize() this operator before dispatching it; it owns " +
                ", ".join(slot.describe() for slot in owned))
        self._initialize(())

    staged = _validate(self, inputs or {}, owned=False, verb="dispatch")
    return self._dispatch(_converted(self, staged))


_core.CompiledOperator.initialize = _operator_initialize
_core.CompiledOperator.dispatch = _operator_dispatch
_core.CompiledOperator.__call__ = _operator_dispatch
