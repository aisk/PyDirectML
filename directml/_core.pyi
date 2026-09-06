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
_ScaleBias = Optional[tuple[float, float]]  # the linear transform on a unary's input

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

class RoundingMode:
    HALVES_TO_NEAREST_EVEN: RoundingMode
    TOWARD_ZERO: RoundingMode
    TOWARD_INFINITY: RoundingMode

class IsInfinityMode:
    EITHER: IsInfinityMode
    POSITIVE: IsInfinityMode
    NEGATIVE: IsInfinityMode

class AxisDirection:
    INCREASING: AxisDirection
    DECREASING: AxisDirection

class DepthSpaceOrder:
    DEPTH_COLUMN_ROW: DepthSpaceOrder
    COLUMN_ROW_DEPTH: DepthSpaceOrder

class ReduceFunction:
    ARGMAX: ReduceFunction
    ARGMIN: ReduceFunction
    AVERAGE: ReduceFunction
    L1: ReduceFunction
    L2: ReduceFunction
    LOG_SUM: ReduceFunction
    LOG_SUM_EXP: ReduceFunction
    MAX: ReduceFunction
    MIN: ReduceFunction
    MULTIPLY: ReduceFunction
    SUM: ReduceFunction
    SUM_SQUARE: ReduceFunction

class RandomGeneratorType:
    PHILOX_4X32_10: RandomGeneratorType

class QuantizationType:
    SCALE: QuantizationType
    SCALE_ZERO_POINT: QuantizationType

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

# Elementwise unary: the same optional (scale, bias) on every one of them.
def identity(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def abs(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def acos(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def asin(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def atan(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def ceil(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def cos(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def exp(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def floor(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def log(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def recip(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def sin(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def sqrt(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def tan(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def erf(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def sinh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def cosh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def tanh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def asinh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def acosh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def atanh(input: Expression, *, scale_bias: _ScaleBias = ...) -> Expression: ...
def sign(input: Expression) -> Expression: ...
def negate(input: Expression) -> Expression: ...
def logical_not(input: Expression) -> Expression: ...
def bit_not(input: Expression) -> Expression: ...
def clip(input: Expression, *, min: float, max: float,
         scale_bias: _ScaleBias = ...) -> Expression: ...
def threshold(input: Expression, *, min: float,
              scale_bias: _ScaleBias = ...) -> Expression: ...
def round(input: Expression, *, rounding_mode: RoundingMode = ...) -> Expression: ...
def is_nan(input: Expression, *, output_dtype: _DtypeLike = ...) -> Expression: ...
def is_infinity(input: Expression, *, infinity_mode: IsInfinityMode = ...,
                output_dtype: _DtypeLike = ...) -> Expression: ...
def bit_count(input: Expression, *, output_dtype: _DtypeLike = ...) -> Expression: ...
def cast(input: Expression, *, dtype: _DtypeLike) -> Expression: ...

# Elementwise binary. Both operands want the same shape and type.
def max(a: Expression, b: Expression) -> Expression: ...
def min(a: Expression, b: Expression) -> Expression: ...
def mean(a: Expression, b: Expression) -> Expression: ...
def atan_yx(a: Expression, b: Expression) -> Expression: ...
def difference_square(a: Expression, b: Expression) -> Expression: ...
def logical_and(a: Expression, b: Expression) -> Expression: ...
def logical_or(a: Expression, b: Expression) -> Expression: ...
def logical_xor(a: Expression, b: Expression) -> Expression: ...
def bit_and(a: Expression, b: Expression) -> Expression: ...
def bit_or(a: Expression, b: Expression) -> Expression: ...
def bit_xor(a: Expression, b: Expression) -> Expression: ...
def bit_shift_left(a: Expression, b: Expression) -> Expression: ...
def bit_shift_right(a: Expression, b: Expression) -> Expression: ...
def modulus_truncate(a: Expression, b: Expression) -> Expression: ...
def modulus_floor(a: Expression, b: Expression) -> Expression: ...
def equals(a: Expression, b: Expression, *,
           output_dtype: _DtypeLike = ...) -> Expression: ...
def greater_than(a: Expression, b: Expression, *,
                 output_dtype: _DtypeLike = ...) -> Expression: ...
def greater_than_or_equal(a: Expression, b: Expression, *,
                          output_dtype: _DtypeLike = ...) -> Expression: ...
def less_than(a: Expression, b: Expression, *,
              output_dtype: _DtypeLike = ...) -> Expression: ...
def less_than_or_equal(a: Expression, b: Expression, *,
                       output_dtype: _DtypeLike = ...) -> Expression: ...
def pow(input: Expression, exponent: Union[Expression, float], *,
        scale_bias: _ScaleBias = ...) -> Expression: ...
def where(condition: Expression, a: Expression, b: Expression) -> Expression: ...

# Activations.
def activation_elu(input: Expression, *, alpha: float = ...) -> Expression: ...
def activation_celu(input: Expression, *, alpha: float = ...) -> Expression: ...
def activation_hardmax(input: Expression) -> Expression: ...
def activation_hard_sigmoid(input: Expression, *, alpha: float = ...,
                            beta: float = ...) -> Expression: ...
def activation_leaky_relu(input: Expression, *, alpha: float = ...) -> Expression: ...
def activation_log_softmax(input: Expression) -> Expression: ...
def activation_parameterized_relu(input: Expression, slope: Expression) -> Expression: ...
def activation_parametric_softplus(input: Expression, *, alpha: float,
                                   beta: float) -> Expression: ...
def activation_scaled_elu(input: Expression, *, alpha: float = ...,
                          gamma: float = ...) -> Expression: ...
def activation_scaled_tanh(input: Expression, *, alpha: float = ...,
                           beta: float = ...) -> Expression: ...
def activation_shrink(input: Expression, *, bias: float = ...,
                      threshold: float = ...) -> Expression: ...
def activation_softplus(input: Expression, *, steepness: float = ...) -> Expression: ...
def activation_softsign(input: Expression) -> Expression: ...
def activation_thresholded_relu(input: Expression, *, alpha: float = ...) -> Expression: ...

# Shape and data movement.
def split(input: Expression, *, axis: int,
          output_axis_sizes: Sequence[int]) -> list[Expression]: ...
def tile(input: Expression, *, repeats: Sequence[int]) -> Expression: ...
def one_hot(indices: Expression, values: Expression, *, output_length: int,
            axis: int) -> Expression: ...
def top_k(input: Expression, *, axis: int, k: int,
          axis_direction: AxisDirection = ...) -> tuple[Expression, Expression]: ...
def gather_elements(input: Expression, indices: Expression, *,
                    axis: int) -> Expression: ...
def gather_nd(input: Expression, indices: Expression, *, input_dimension_count: int,
              indices_dimension_count: int,
              batch_dimension_count: int = ...) -> Expression: ...
def scatter_elements(input: Expression, indices: Expression, updates: Expression, *,
                     axis: int) -> Expression: ...
def scatter_nd(input: Expression, indices: Expression, updates: Expression, *,
               input_dimension_count: int,
               indices_dimension_count: int) -> Expression: ...
def space_to_depth(input: Expression, *, block_size: int,
                   order: DepthSpaceOrder = ...) -> Expression: ...
def depth_to_space(input: Expression, *, block_size: int,
                   order: DepthSpaceOrder = ...) -> Expression: ...
def reverse_subsequences(input: Expression, sequence_lengths: Expression, *,
                         axis: int) -> Expression: ...
def resample(input: Expression, *, output_sizes: _Shape, mode: InterpolationMode,
             rounding_direction: AxisDirection = ...,
             scales: Sequence[float] = ...,
             input_pixel_offsets: Sequence[float] = ...,
             output_pixel_offsets: Sequence[float] = ...,
             antialiased: bool = ...) -> Expression: ...
def fill_value_constant(graph: Graph, *, sizes: _Shape, value: float,
                        dtype: Optional[_DtypeLike] = ...) -> Expression: ...
def fill_value_sequence(graph: Graph, *, sizes: _Shape, value_start: float,
                        value_delta: float,
                        dtype: Optional[_DtypeLike] = ...) -> Expression: ...

# Reductions, and the operators whose output is an answer about the input.
def reduce(input: Expression, *, function: ReduceFunction, axes: Sequence[int] = ...,
           output_dtype: Optional[_DtypeLike] = ...) -> Expression: ...
def cumulative_summation(input: Expression, *, axis: int,
                         axis_direction: AxisDirection = ...,
                         has_exclusive_sum: bool = ...) -> Expression: ...
def cumulative_product(input: Expression, *, axis: int,
                       axis_direction: AxisDirection = ...,
                       has_exclusive_product: bool = ...) -> Expression: ...
def non_zero_coordinates(input: Expression) -> tuple[Expression, Expression]: ...
def quantize_linear(input: Expression, scale: Expression, zero_point: Expression, *,
                    output_dtype: _DtypeLike = ...) -> Expression: ...
def dequantize_linear(input: Expression, scale: Expression,
                      zero_point: Expression) -> Expression: ...
def dequantize(input: Expression, quantization_parameters: Sequence[Expression],
               *, quantization_type: QuantizationType) -> Expression: ...
def random_generator(input_state: Expression, *, output_sizes: _Shape,
                     output_state: bool = ...,
                     type: RandomGeneratorType = ...) -> tuple[Expression, Optional[Expression]]: ...
def roi_align(
    input: Expression,
    roi: Expression,
    batch_indices: Expression,
    *,
    reduction_function: ReduceFunction,
    interpolation_mode: InterpolationMode,
    spatial_scale_x: float,
    spatial_scale_y: float,
    input_pixel_offset: float,
    output_pixel_offset: float,
    out_of_bounds_input_value: float,
    minimum_samples_per_output: int,
    maximum_samples_per_output: int,
    align_regions_to_corners: bool,
    output_height: int,
    output_width: int,
) -> Expression: ...

# Quantized convolution.
def convolution_integer(
    input: Expression,
    filter: Expression,
    input_zero_point: Optional[Expression] = ...,
    filter_zero_point: Optional[Expression] = ...,
    *,
    strides: Sequence[int] = ...,
    dilations: Sequence[int] = ...,
    start_padding: Sequence[int] = ...,
    end_padding: Sequence[int] = ...,
    group_count: int = ...,
    output_sizes: _Shape = ...,
) -> Expression: ...
def quantized_linear_convolution(
    input: Expression,
    input_scale: Expression,
    filter: Expression,
    filter_scale: Expression,
    output_scale: Expression,
    input_zero_point: Optional[Expression] = ...,
    filter_zero_point: Optional[Expression] = ...,
    bias: Optional[Expression] = ...,
    output_zero_point: Optional[Expression] = ...,
    *,
    output_dtype: _DtypeLike,
    strides: Sequence[int] = ...,
    dilations: Sequence[int] = ...,
    start_padding: Sequence[int] = ...,
    end_padding: Sequence[int] = ...,
    group_count: int = ...,
    output_sizes: _Shape = ...,
) -> Expression: ...

# Gradients.
def clip_grad(input: Expression, input_gradient: Expression, *, min: float,
              max: float) -> Expression: ...
def batch_normalization_grad(
    input: Expression,
    input_gradient: Expression,
    mean: Expression,
    variance: Expression,
    scale: Expression,
    *,
    epsilon: float = ...,
) -> tuple[Expression, Expression, Expression]: ...
def batch_normalization_training(
    input: Expression,
    scale: Expression,
    bias: Expression,
    fused_add: Optional[Expression] = ...,
    *,
    epsilon: float = ...,
    fused_activation: Optional[FusedActivation] = ...,
) -> tuple[Expression, Expression, Expression]: ...
def batch_normalization_training_grad(
    input: Expression,
    input_gradient: Expression,
    mean: Expression,
    variance: Expression,
    scale: Expression,
    *,
    epsilon: float = ...,
) -> tuple[Expression, Expression, Expression]: ...
def resample_grad(input_gradient: Expression, *, output_sizes: _Shape,
                  mode: InterpolationMode,
                  scales: Sequence[float] = ...,
                  input_pixel_offsets: Sequence[float] = ...,
                  output_pixel_offsets: Sequence[float] = ...) -> Expression: ...
def slice_grad(input_gradient: Expression, *, output_gradient_sizes: _Shape,
               input_window_offsets: Sequence[int],
               input_window_sizes: Sequence[int],
               input_window_strides: Sequence[int]) -> Expression: ...
def roi_align_grad(
    input_gradient: Expression,
    roi: Expression,
    batch_indices: Expression,
    input: Optional[Expression] = ...,
    *,
    reduction_function: ReduceFunction,
    interpolation_mode: InterpolationMode,
    spatial_scale_x: float,
    spatial_scale_y: float,
    input_pixel_offset: float,
    output_pixel_offset: float,
    minimum_samples_per_output: int,
    maximum_samples_per_output: int,
    align_regions_to_corners: bool,
    batch_size: int,
    image_height: int,
    image_width: int,
    compute_output_gradient: bool = ...,
    compute_output_roi_gradient: bool = ...,
) -> tuple[Optional[Expression], Optional[Expression]]: ...
