"""Type stub for the compiled core.

The wrapper layer in __init__.py is ordinary Python and documents itself;
only this extension module needs a stub. It also declares the methods and
properties the wrapper attaches to these classes at import time
(Expression.shape, Graph.input, CompiledOperator.initialize, ...), because
that is the surface an instance actually has.
"""

from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

_Shape = Sequence[int]
_Strides = Optional[Sequence[int]]
_DtypeLike = Union[TensorDataType, np.dtype, type]
_Array = Any  # anything np.asarray accepts
_Data = Union[_Array, Buffer]  # what a binding dict maps an input to
_Key = Union[Expression, str]  # an input, or the name it was given

# --- Enumerations ------------------------------------------------------------

class TensorDataType:
    UNKNOWN: TensorDataType
    FLOAT32: TensorDataType
    FLOAT16: TensorDataType
    UINT32: TensorDataType
    UINT16: TensorDataType
    UINT8: TensorDataType
    INT32: TensorDataType
    INT16: TensorDataType
    INT8: TensorDataType
    FLOAT64: TensorDataType
    UINT64: TensorDataType
    INT64: TensorDataType

class TensorFlags:
    NONE: TensorFlags
    OWNED_BY_DML: TensorFlags

class MatrixTransform:
    NONE: MatrixTransform
    TRANSPOSE: MatrixTransform

class RecurrentNetworkDirection:
    FORWARD: RecurrentNetworkDirection
    BACKWARD: RecurrentNetworkDirection
    BIDIRECTIONAL: RecurrentNetworkDirection

class GRUOutputOptions:
    Both: GRUOutputOptions
    Sequence: GRUOutputOptions
    Single: GRUOutputOptions

class ConvolutionMode:
    CONVOLUTION: ConvolutionMode
    CROSS_CORRELATION: ConvolutionMode

class ConvolutionDirection:
    FORWARD: ConvolutionDirection
    BACKWARD: ConvolutionDirection

class InterpolationMode:
    NEAREST_NEIGHBOR: InterpolationMode
    LINEAR: InterpolationMode

class PaddingMode:
    CONSTANT: PaddingMode
    EDGE: PaddingMode
    REFLECTION: PaddingMode

class ExecutionFlags:
    NONE: ExecutionFlags
    ALLOW_HALF_PRECISION_COMPUTATION: ExecutionFlags
    DISABLE_META_COMMANDS: ExecutionFlags
    DESCRIPTORS_VOLATILE: ExecutionFlags

class OperatorType:
    # Only the members the samples and factories reach for are listed by name;
    # the enum carries one member per DML_OPERATOR_TYPE value bound in
    # module.cpp.
    INVALID: OperatorType
    ACTIVATION_ELU: OperatorType
    ACTIVATION_HARD_SIGMOID: OperatorType
    ACTIVATION_IDENTITY: OperatorType
    ACTIVATION_LEAKY_RELU: OperatorType
    ACTIVATION_LINEAR: OperatorType
    ACTIVATION_PARAMETRIC_SOFTPLUS: OperatorType
    ACTIVATION_RELU: OperatorType
    ACTIVATION_SCALED_ELU: OperatorType
    ACTIVATION_SCALED_TANH: OperatorType
    ACTIVATION_SIGMOID: OperatorType
    ACTIVATION_SOFTPLUS: OperatorType
    ACTIVATION_SOFTSIGN: OperatorType
    ACTIVATION_TANH: OperatorType
    ACTIVATION_THRESHOLDED_RELU: OperatorType
    ACTIVATION_SHRINK: OperatorType
    ACTIVATION_CELU: OperatorType
    ACTIVATION_GELU: OperatorType
    def __getattr__(self, name: str) -> OperatorType: ...

# --- Classes -----------------------------------------------------------------

class TensorPolicy:
    default: TensorPolicy
    interleaved_channel: TensorPolicy

class TensorDesc:
    # __init__ is replaced by the wrapper layer, which also accepts numpy dtypes.
    def __init__(
        self,
        data_type: _DtypeLike,
        sizes: _Shape,
        *,
        flags: TensorFlags = ...,
        strides: _Strides = ...,
        total_tensor_size_in_bytes: Optional[int] = ...,
        guaranteed_base_offset_alignment: int = ...,
        tensor_policy: Optional[TensorPolicy] = ...,
    ) -> None: ...
    @property
    def data_type(self) -> TensorDataType: ...
    @property
    def flags(self) -> TensorFlags: ...
    @property
    def sizes(self) -> list[int]: ...
    @property
    def strides(self) -> Optional[list[int]]: ...
    @property
    def total_tensor_size_in_bytes(self) -> int: ...
    @property
    def guaranteed_base_offset_alignment(self) -> int: ...

class Expression:
    @property
    def desc(self) -> TensorDesc: ...
    # Attached by the wrapper layer:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def strides(self) -> Optional[tuple[int, ...]]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def size(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __add__(self, other: Union[Expression, float]) -> Expression: ...
    def __sub__(self, other: Union[Expression, float]) -> Expression: ...
    def __mul__(self, other: Union[Expression, float]) -> Expression: ...
    def __truediv__(self, other: Union[Expression, float]) -> Expression: ...
    def __mod__(self, other: Expression) -> Expression: ...
    def __radd__(self, other: float) -> Expression: ...
    def __rsub__(self, other: float) -> Expression: ...
    def __rmul__(self, other: float) -> Expression: ...
    def __rtruediv__(self, other: float) -> Expression: ...
    def __neg__(self) -> Expression: ...

class FusedActivation:
    def __init__(self, activation: OperatorType, param_1: float = ...,
                 param_2: float = ...) -> None: ...
    # Attached by the wrapper layer:
    @staticmethod
    def none() -> FusedActivation: ...
    @staticmethod
    def elu(alpha: float = ...) -> FusedActivation: ...
    @staticmethod
    def hard_sigmoid(alpha: float = ..., beta: float = ...) -> FusedActivation: ...
    @staticmethod
    def identity() -> FusedActivation: ...
    @staticmethod
    def leaky_relu(alpha: float = ...) -> FusedActivation: ...
    @staticmethod
    def linear(alpha: float, beta: float) -> FusedActivation: ...
    @staticmethod
    def parametric_softplus(alpha: float, beta: float) -> FusedActivation: ...
    @staticmethod
    def relu() -> FusedActivation: ...
    @staticmethod
    def scaled_elu(alpha: float = ..., gamma: float = ...) -> FusedActivation: ...
    @staticmethod
    def scaled_tanh(alpha: float = ..., beta: float = ...) -> FusedActivation: ...
    @staticmethod
    def sigmoid() -> FusedActivation: ...
    @staticmethod
    def softplus(steepness: float = ...) -> FusedActivation: ...
    @staticmethod
    def softsign() -> FusedActivation: ...
    @staticmethod
    def tanh() -> FusedActivation: ...
    @staticmethod
    def thresholded_relu(alpha: float = ...) -> FusedActivation: ...
    @staticmethod
    def shrink(bias: float = ..., threshold: float = ...) -> FusedActivation: ...
    @staticmethod
    def celu(alpha: float = ...) -> FusedActivation: ...
    @staticmethod
    def gelu() -> FusedActivation: ...

class Device:
    def __init__(self, *, use_gpu: bool = ..., use_debug_layer: bool = ...) -> None: ...

class Buffer:
    # __init__ is replaced by the wrapper layer, which converts the array.
    def __init__(self, device: Device, array: _Array, dtype: Optional[_DtypeLike] = ...) -> None: ...
    @property
    def desc(self) -> TensorDesc: ...
    @property
    def device(self) -> Device: ...
    @property
    def nbytes(self) -> int: ...
    def numpy(self) -> np.ndarray: ...
    # Attached by the wrapper layer:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def strides(self) -> Optional[tuple[int, ...]]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def size(self) -> int: ...

class CompiledOperator:
    @property
    def temporary_size(self) -> int: ...
    @property
    def persistent_size(self) -> int: ...
    @property
    def descriptor_count(self) -> int: ...
    @property
    def initialized(self) -> bool: ...
    # Attached by the wrapper layer:
    def initialize(self, weights: Optional[dict[_Key, _Data]] = ...) -> None: ...
    def dispatch(self, inputs: Optional[dict[_Key, _Data]] = ..., *,
                 readback: bool = ...) -> list[Union[np.ndarray, Buffer]]: ...
    def __call__(self, inputs: Optional[dict[_Key, _Data]] = ..., *,
                 readback: bool = ...) -> list[Union[np.ndarray, Buffer]]: ...

class Graph:
    def __init__(self, device: Device, *,
                 tensor_policy: Optional[TensorPolicy] = ...) -> None: ...
    # Attached by the wrapper layer:
    def input(self, sizes: Optional[_Shape] = ..., dtype: Optional[_DtypeLike] = ...,
              *, owned: bool = ..., strides: _Strides = ...,
              desc: Optional[TensorDesc] = ..., name: Optional[str] = ...) -> Expression: ...
    def constant(self, array: _Data, dtype: Optional[_DtypeLike] = ..., *,
                 sizes: Optional[_Shape] = ..., name: Optional[str] = ...) -> Expression: ...
    def compile(self, outputs: Sequence[Expression], *,
                flags: ExecutionFlags = ...) -> CompiledOperator: ...

# --- Operators ---------------------------------------------------------------

def convolution(
    input: Expression,
    filter: Expression,
    bias: Optional[Expression] = ...,
    *,
    mode: ConvolutionMode = ...,
    direction: ConvolutionDirection = ...,
    strides: Sequence[int] = ...,
    dilations: Sequence[int] = ...,
    start_padding: Sequence[int] = ...,
    end_padding: Sequence[int] = ...,
    output_padding: Sequence[int] = ...,
    group_count: int = ...,
    fused_activation: Optional[FusedActivation] = ...,
    output_sizes: Sequence[int] = ...,
) -> Expression: ...
def upsample_2d(input: Expression, *, scale_size: tuple[int, int],
                interpolation_mode: InterpolationMode) -> Expression: ...
def activation_relu(input: Expression) -> Expression: ...
def activation_sigmoid(input: Expression) -> Expression: ...
def activation_identity(input: Expression) -> Expression: ...
def activation_tanh(input: Expression) -> Expression: ...
def activation_gelu(input: Expression) -> Expression: ...
def activation_linear(input: Expression, *, alpha: float, beta: float) -> Expression: ...
def activation_softmax(input: Expression, *, axes: Sequence[int] = ...) -> Expression: ...
def add(a: Expression, b: Expression, *,
        fused_activation: Optional[FusedActivation] = ...) -> Expression: ...
def subtract(a: Expression, b: Expression) -> Expression: ...
def multiply(a: Expression, b: Expression) -> Expression: ...
def divide(a: Expression, b: Expression) -> Expression: ...
def padding(input: Expression, *, padding_mode: PaddingMode = ...,
            padding_value: float = ..., start_padding: Sequence[int],
            end_padding: Sequence[int]) -> Expression: ...
def mean_variance_normalization(
    input: Expression,
    scale: Optional[Expression] = ...,
    bias: Optional[Expression] = ...,
    *,
    axes: Sequence[int],
    normalize_variance: bool = ...,
    normalize_mean: bool = ...,
    epsilon: float = ...,
    fused_activation: Optional[FusedActivation] = ...,
) -> Expression: ...
def slice(input: Expression, *, input_window_offsets: Sequence[int],
          input_window_sizes: Sequence[int],
          input_window_strides: Sequence[int]) -> Expression: ...
def value_scale_2d(input: Expression, *, scale: float,
                   bias: Sequence[float]) -> Expression: ...
def batch_normalization(
    input: Expression,
    mean: Expression,
    variance: Expression,
    scale: Expression,
    bias: Expression,
    *,
    spatial: bool = ...,
    epsilon: float = ...,
    fused_activation: Optional[FusedActivation] = ...,
) -> Expression: ...
def local_response_normalization(input: Expression, *, cross_channel: bool,
                                 local_size: int, alpha: float, beta: float,
                                 bias: float) -> Expression: ...
def gemm(
    a: Expression,
    b: Expression,
    c: Optional[Expression] = ...,
    *,
    trans_a: MatrixTransform = ...,
    trans_b: MatrixTransform = ...,
    alpha: float = ...,
    beta: float = ...,
    fused_activation: Optional[FusedActivation] = ...,
) -> Expression: ...
def average_pooling(
    input: Expression,
    *,
    strides: Sequence[int],
    window_sizes: Sequence[int],
    start_padding: Sequence[int] = ...,
    end_padding: Sequence[int] = ...,
    dilations: Sequence[int] = ...,
    include_padding: bool = ...,
    output_sizes: Sequence[int] = ...,
) -> Expression: ...
def max_pooling(
    input: Expression,
    *,
    window_sizes: Sequence[int],
    strides: Sequence[int] = ...,
    start_padding: Sequence[int] = ...,
    end_padding: Sequence[int] = ...,
    dilations: Sequence[int] = ...,
    output_indices: bool = ...,
) -> tuple[Expression, Optional[Expression]]: ...
def reinterpret(input: Expression, sizes: Sequence[int],
                strides: _Strides = ...,
                dtype: Optional[_DtypeLike] = ...) -> Expression: ...
def broadcast(input: Expression, shape: Sequence[int]) -> Expression: ...
def multihead_attention(query: Expression, key: Expression, value: Expression,
                        *, head_count: int, scale: float) -> Expression: ...
def join(inputs: Sequence[Expression], *, axis: int) -> Expression: ...
def gru(
    input: Expression,
    weight: Expression,
    recurrence: Expression,
    bias: Optional[Expression] = ...,
    hidden_init: Optional[Expression] = ...,
    sequence_lengths: Optional[Expression] = ...,
    *,
    activation_descs: Sequence[FusedActivation],
    direction: RecurrentNetworkDirection = ...,
    linear_before_reset: bool = ...,
    output_options: GRUOutputOptions = ...,
) -> tuple[Optional[Expression], Optional[Expression]]: ...
def gather(input: Expression, indices: Expression, *, axis: int,
           index_dimensions: int) -> Expression: ...
