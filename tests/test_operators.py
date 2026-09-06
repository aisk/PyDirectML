"""Every operator, against numpy on a small tensor.

One graph, one operator, one dispatch: enough to catch a binding that passes
its arguments in the wrong order or reads the wrong output, which is the
mistake a signature copied from DirectMLX invites. The numerics are DirectML's
and are not what is under test here.
"""

import math

import numpy as np
import pytest

import directml as dml


@pytest.fixture(scope="module")
def device():
    return dml.Device()


def compute(device, build, *arrays):
    """Dispatch a one-operator graph over the arrays and return its output."""
    return compute_all(device, build, *arrays)[0]


def compute_all(device, build, *arrays):
    """The same, for an operator with more than one output."""
    graph = dml.Graph(device)
    inputs = [graph.input(a.shape, a.dtype) for a in arrays]
    outputs = build(*inputs)
    if isinstance(outputs, dml.Expression):
        outputs = [outputs]
    outputs = [o for o in outputs if o is not None]
    return graph.compile(outputs)(dict(zip(inputs, arrays)))


def shaped(values, dtype=np.float32):
    """``values`` as a 4-D tensor, which is the rank DirectML operators want."""
    array = np.asarray(values, dtype)
    return array.reshape((1,) * (4 - array.ndim) + array.shape)


# Inputs picked for the domain of the function under test.
X = shaped([[-2.0, -0.5], [0.5, 2.0]])
UNIT = shaped([[-0.75, -0.25], [0.25, 0.75]])
POSITIVE = shaped([[0.25, 0.5], [1.5, 4.0]])
ABOVE_ONE = shaped([[1.25, 1.5], [2.0, 4.0]])
A = shaped([[1.0, 2.0], [3.0, 4.0]])
B = shaped([[4.0, 3.0], [2.0, 1.0]])
BITS = shaped([[1, 6], [255, 4096]], np.uint32)
ROW = shaped([1.0, 3.0, 2.0, 0.5])


class TestElementwiseUnary:
    """The unary operators, and the (scale, bias) every one of them folds into
    the read of its input."""

    @pytest.mark.parametrize("name, reference, array", [
        ("identity", lambda a: a, X),
        ("abs", np.abs, X),
        ("acos", np.arccos, UNIT),
        ("asin", np.arcsin, UNIT),
        ("atan", np.arctan, X),
        ("ceil", np.ceil, X),
        ("cos", np.cos, X),
        ("exp", np.exp, X),
        ("floor", np.floor, X),
        ("log", np.log, POSITIVE),
        ("recip", np.reciprocal, X),
        ("sin", np.sin, X),
        ("sqrt", np.sqrt, POSITIVE),
        ("tan", np.tan, UNIT),
        ("erf", np.vectorize(math.erf), X),
        ("sinh", np.sinh, X),
        ("cosh", np.cosh, X),
        ("tanh", np.tanh, X),
        ("asinh", np.arcsinh, X),
        ("acosh", np.arccosh, ABOVE_ONE),
        ("atanh", np.arctanh, UNIT),
        ("sign", np.sign, X),
        ("negate", np.negative, X),
    ])
    def test_matches_numpy(self, device, name, reference, array):
        operator = getattr(dml, name)
        result = compute(device, lambda x: operator(x), array)
        assert np.allclose(result, reference(array), rtol=1e-4, atol=1e-5)

    def test_scale_and_bias_are_applied_to_the_input(self, device):
        result = compute(device, lambda x: dml.abs(x, scale_bias=(2.0, 1.0)), X)
        assert np.allclose(result, np.abs(X * 2.0 + 1.0))

    def test_clip(self, device):
        result = compute(device, lambda x: dml.clip(x, min=-1.0, max=1.0), X)
        assert np.allclose(result, np.clip(X, -1.0, 1.0))

    def test_threshold_is_the_one_sided_clip(self, device):
        result = compute(device, lambda x: dml.threshold(x, min=0.0), X)
        assert np.allclose(result, np.maximum(X, 0.0))

    @pytest.mark.parametrize("mode, reference", [
        (dml.RoundingMode.HALVES_TO_NEAREST_EVEN, [0.0, 2.0, -2.0, 2.0]),
        (dml.RoundingMode.TOWARD_ZERO, [0.0, 1.0, -1.0, 2.0]),
        (dml.RoundingMode.TOWARD_INFINITY, [1.0, 2.0, -2.0, 2.0]),
    ])
    def test_round(self, device, mode, reference):
        array = shaped([0.5, 1.5, -1.5, 2.4])
        result = compute(device, lambda x: dml.round(x, rounding_mode=mode), array)
        assert np.allclose(result.ravel(), reference)

    def test_is_nan(self, device):
        array = shaped([0.0, np.nan, np.inf, np.nan])
        result = compute(device, dml.is_nan, array)
        assert result.dtype == np.uint8
        assert np.array_equal(result.ravel(), [0, 1, 0, 1])

    @pytest.mark.parametrize("mode, expected", [
        (dml.IsInfinityMode.EITHER, [0, 1, 1, 0]),
        (dml.IsInfinityMode.POSITIVE, [0, 1, 0, 0]),
        (dml.IsInfinityMode.NEGATIVE, [0, 0, 1, 0]),
    ])
    def test_is_infinity(self, device, mode, expected):
        array = shaped([1.0, np.inf, -np.inf, np.nan])
        result = compute(device, lambda x: dml.is_infinity(x, infinity_mode=mode), array)
        assert np.array_equal(result.ravel(), expected)

    def test_output_dtype_takes_either_spelling(self, device):
        # uint8 and uint32 are the two types DirectML writes a predicate as.
        array = shaped([0.0, np.nan, 0.0, np.nan])
        result = compute(device, lambda x: dml.is_nan(x, output_dtype=np.uint32), array)
        assert result.dtype == np.uint32
        assert np.array_equal(result.ravel(), [0, 1, 0, 1])

    def test_logical_not(self, device):
        # The logical operators read and write a predicate, so uint8 or uint32.
        array = shaped([0, 1, 0, 2], np.uint8)
        result = compute(device, dml.logical_not, array)
        assert np.array_equal(result.ravel(), [1, 0, 1, 0])

    def test_bit_not(self, device):
        result = compute(device, dml.bit_not, BITS)
        assert np.array_equal(result, ~BITS)

    def test_bit_count(self, device):
        result = compute(device, dml.bit_count, BITS)
        assert result.dtype == np.uint8
        assert np.array_equal(result.ravel(), [1, 2, 8, 1])

    def test_cast_truncates_towards_zero(self, device):
        array = shaped([1.7, -1.7, 2.5, -0.5])
        result = compute(device, lambda x: dml.cast(x, dtype=np.int32), array)
        assert result.dtype == np.int32
        assert np.array_equal(result.ravel(), [1, -1, 2, 0])


class TestElementwiseBinary:
    @pytest.mark.parametrize("name, reference", [
        ("max", np.maximum),
        ("min", np.minimum),
        ("mean", lambda a, b: (a + b) / 2),
        ("difference_square", lambda a, b: (a - b) ** 2),
        ("atan_yx", np.arctan2),
    ])
    def test_matches_numpy(self, device, name, reference):
        operator = getattr(dml, name)
        result = compute(device, lambda a, b: operator(a, b), A, B)
        assert np.allclose(result, reference(A, B), rtol=1e-4, atol=1e-5)

    def test_pow_of_a_tensor(self, device):
        result = compute(device, dml.pow, A, B)
        assert np.allclose(result, A ** B, rtol=1e-4)

    def test_pow_of_a_constant(self, device):
        result = compute(device, lambda x: dml.pow(x, 2.0), A)
        assert np.allclose(result, A ** 2)

    @pytest.mark.parametrize("name, reference", [
        ("logical_and", lambda a, b: (a != 0) & (b != 0)),
        ("logical_or", lambda a, b: (a != 0) | (b != 0)),
        ("logical_xor", lambda a, b: (a != 0) ^ (b != 0)),
    ])
    def test_logical(self, device, name, reference):
        a = shaped([0, 0, 1, 2], np.uint8)
        b = shaped([0, 3, 0, 4], np.uint8)
        operator = getattr(dml, name)
        result = compute(device, lambda x, y: operator(x, y), a, b)
        assert np.array_equal(result.ravel(), reference(a, b).ravel())

    @pytest.mark.parametrize("name, reference", [
        ("equals", np.equal),
        ("greater_than", np.greater),
        ("greater_than_or_equal", np.greater_equal),
        ("less_than", np.less),
        ("less_than_or_equal", np.less_equal),
    ])
    def test_comparisons(self, device, name, reference):
        a = shaped([1.0, 2.0, 3.0, 4.0])
        b = shaped([4.0, 2.0, 2.0, 1.0])
        operator = getattr(dml, name)
        result = compute(device, lambda x, y: operator(x, y), a, b)
        assert result.dtype == np.uint8
        assert np.array_equal(result.ravel(), reference(a, b).ravel())

    def test_comparison_writes_the_type_asked_for(self, device):
        a = shaped([1.0, 2.0, 3.0, 4.0])
        b = shaped([4.0, 2.0, 2.0, 1.0])
        result = compute(device, lambda x, y: dml.less_than(x, y, output_dtype=np.uint32), a, b)
        assert result.dtype == np.uint32

    @pytest.mark.parametrize("name, reference", [
        ("bit_and", lambda a, b: a & b),
        ("bit_or", lambda a, b: a | b),
        ("bit_xor", lambda a, b: a ^ b),
    ])
    def test_bitwise(self, device, name, reference):
        other = shaped([[3, 5], [15, 4096]], np.uint32)
        operator = getattr(dml, name)
        result = compute(device, lambda a, b: operator(a, b), BITS, other)
        assert np.array_equal(result, reference(BITS, other))

    def test_bit_shifts(self, device):
        counts = shaped([[1, 2], [3, 4]], np.uint32)
        left = compute(device, dml.bit_shift_left, BITS, counts)
        right = compute(device, dml.bit_shift_right, BITS, counts)
        assert np.array_equal(left, BITS << counts)
        assert np.array_equal(right, BITS >> counts)

    def test_modulus_keeps_the_sign_it_says_it_does(self, device):
        a = shaped([-7, -1, 1, 7], np.int32)
        b = shaped([5, 5, -5, -5], np.int32)
        truncated = compute(device, dml.modulus_truncate, a, b)
        floored = compute(device, dml.modulus_floor, a, b)
        assert np.array_equal(truncated.ravel(), [-2, -1, 1, 2])       # C's %
        assert np.array_equal(floored.ravel(), (a % b).ravel())        # Python's

    def test_where_selects_elementwise(self, device):
        condition = shaped([1, 0, 1, 0], np.uint8)
        a = shaped([1.0, 2.0, 3.0, 4.0])
        b = shaped([5.0, 6.0, 7.0, 8.0])
        result = compute(device, dml.where, condition, a, b)
        assert np.array_equal(result.ravel(), [1.0, 6.0, 3.0, 8.0])

    def test_mismatched_operands_name_the_call(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2])
        b = graph.input([1, 1, 2, 3])
        with pytest.raises(ValueError, match=r"max\(.*shapes differ"):
            dml.max(a, b)
        with pytest.raises(ValueError, match=r"where\(.*shapes differ"):
            dml.where(a, a, b)


class TestActivations:
    def test_elu(self, device):
        result = compute(device, lambda x: dml.activation_elu(x, alpha=2.0), X)
        assert np.allclose(result, np.where(X > 0, X, 2.0 * (np.exp(X) - 1)))

    def test_celu(self, device):
        alpha = 2.0
        result = compute(device, lambda x: dml.activation_celu(x, alpha=alpha), X)
        expected = np.maximum(0, X) + np.minimum(0, alpha * (np.exp(X / alpha) - 1))
        assert np.allclose(result, expected, rtol=1e-4)

    def test_hardmax(self, device):
        result = compute(device, dml.activation_hardmax, ROW)
        assert np.array_equal(result.ravel(), [0.0, 1.0, 0.0, 0.0])

    def test_hard_sigmoid(self, device):
        result = compute(device, lambda x: dml.activation_hard_sigmoid(x), X)
        assert np.allclose(result, np.clip(0.2 * X + 0.5, 0, 1))

    def test_leaky_relu(self, device):
        result = compute(device, lambda x: dml.activation_leaky_relu(x, alpha=0.1), X)
        assert np.allclose(result, np.where(X > 0, X, 0.1 * X))

    def test_log_softmax(self, device):
        result = compute(device, dml.activation_log_softmax, ROW)
        expected = ROW - np.log(np.exp(ROW).sum())
        assert np.allclose(result, expected, rtol=1e-4)

    def test_parameterized_relu(self, device):
        slope = shaped([[0.1, 0.2], [0.3, 0.4]])
        result = compute(device, dml.activation_parameterized_relu, X, slope)
        assert np.allclose(result, np.where(X > 0, X, slope * X))

    def test_parametric_softplus(self, device):
        result = compute(
            device, lambda x: dml.activation_parametric_softplus(x, alpha=2.0, beta=0.5), X)
        assert np.allclose(result, 2.0 * np.log(1 + np.exp(0.5 * X)), rtol=1e-4)

    def test_scaled_elu(self, device):
        alpha, gamma = 1.5, 2.0
        result = compute(
            device, lambda x: dml.activation_scaled_elu(x, alpha=alpha, gamma=gamma), X)
        expected = gamma * np.where(X > 0, X, alpha * (np.exp(X) - 1))
        assert np.allclose(result, expected, rtol=1e-4)

    def test_scaled_tanh(self, device):
        result = compute(
            device, lambda x: dml.activation_scaled_tanh(x, alpha=2.0, beta=0.5), X)
        assert np.allclose(result, 2.0 * np.tanh(0.5 * X), rtol=1e-4)

    def test_shrink(self, device):
        array = shaped([-2.0, -0.25, 0.25, 2.0])
        result = compute(
            device, lambda x: dml.activation_shrink(x, bias=0.5, threshold=1.0), array)
        assert np.allclose(result.ravel(), [-1.5, 0.0, 0.0, 1.5])

    def test_softplus(self, device):
        result = compute(device, lambda x: dml.activation_softplus(x, steepness=2.0), X)
        assert np.allclose(result, np.log(1 + np.exp(2.0 * X)) / 2.0, rtol=1e-4)

    def test_softsign(self, device):
        result = compute(device, dml.activation_softsign, X)
        assert np.allclose(result, X / (1 + np.abs(X)))

    def test_thresholded_relu(self, device):
        result = compute(
            device, lambda x: dml.activation_thresholded_relu(x, alpha=1.0), X)
        assert np.allclose(result, np.where(X > 1.0, X, 0.0))


class TestShape:
    def test_split(self, device):
        array = shaped([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        left, right = compute_all(
            device, lambda x: dml.split(x, axis=3, output_axis_sizes=[1, 2]), array)
        assert np.array_equal(left.ravel(), [1.0, 4.0])
        assert np.array_equal(right.ravel(), [2.0, 3.0, 5.0, 6.0])

    def test_tile(self, device):
        result = compute(device, lambda x: dml.tile(x, repeats=[1, 1, 2, 1]), A)
        assert result.shape == (1, 1, 4, 2)
        assert np.array_equal(result, np.tile(A, (1, 1, 2, 1)))

    def test_one_hot(self, device):
        # The active axis is 1 wide on the way in and output_length wide out.
        indices = shaped([[0], [2]], np.uint32)
        values = shaped([0.0, 1.0])
        result = compute(
            device, lambda i, v: dml.one_hot(i, v, output_length=4, axis=3),
            indices, values)
        assert result.shape == (1, 1, 2, 4)
        assert np.array_equal(result[0, 0], np.eye(4, dtype=np.float32)[[0, 2]])

    def test_top_k(self, device):
        outputs = dml.TopKOutputs(*[None, None])
        graph = dml.Graph(device)
        x = graph.input(ROW.shape, ROW.dtype)
        outputs = dml.top_k(x, axis=3, k=2)
        values, indices = graph.compile(list(outputs))({x: ROW})
        assert np.array_equal(values.ravel(), [3.0, 2.0])
        assert np.array_equal(indices.ravel(), [1, 2])

    def test_top_k_can_take_the_smallest(self, device):
        graph = dml.Graph(device)
        x = graph.input(ROW.shape, ROW.dtype)
        outputs = dml.top_k(x, axis=3, k=2, axis_direction=dml.AxisDirection.INCREASING)
        values, _ = graph.compile(list(outputs))({x: ROW})
        assert np.array_equal(values.ravel(), [0.5, 1.0])

    def test_gather_elements(self, device):
        indices = shaped([[1, 0], [0, 1]], np.uint32)
        result = compute(
            device, lambda x, i: dml.gather_elements(x, i, axis=3), A, indices)
        assert np.array_equal(result[0, 0], [[2.0, 1.0], [3.0, 4.0]])

    def test_gather_nd(self, device):
        # Three (row, column) coordinates into the last two axes.
        coordinates = shaped([[0, 0], [1, 1], [1, 0]], np.uint32)
        result = compute(
            device,
            lambda x, i: dml.gather_nd(x, i, input_dimension_count=2,
                                       indices_dimension_count=2),
            A, coordinates)
        assert np.array_equal(result.ravel(), [1.0, 4.0, 3.0])

    def test_scatter_elements(self, device):
        indices = shaped([[1, 0], [0, 1]], np.uint32)
        updates = shaped([[10.0, 20.0], [30.0, 40.0]])
        result = compute(
            device,
            lambda x, i, u: dml.scatter_elements(x, i, u, axis=3),
            A, indices, updates)
        assert np.array_equal(result[0, 0], [[20.0, 10.0], [30.0, 40.0]])

    def test_scatter_nd(self, device):
        coordinates = shaped([[0, 0], [1, 1]], np.uint32)
        updates = shaped([10.0, 40.0])
        result = compute(
            device,
            lambda x, i, u: dml.scatter_nd(x, i, u, input_dimension_count=2,
                                           indices_dimension_count=2),
            A, coordinates, updates)
        assert np.array_equal(result[0, 0], [[10.0, 2.0], [3.0, 40.0]])

    def test_space_to_depth_and_back(self, device):
        array = shaped(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
        packed = compute(device, lambda x: dml.space_to_depth(x, block_size=2), array)
        assert packed.shape == (1, 4, 2, 2)
        unpacked = compute(device, lambda x: dml.depth_to_space(x, block_size=2), packed)
        assert np.array_equal(unpacked, array)

    def test_reverse_subsequences(self, device):
        array = shaped([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        lengths = shaped([[2], [4]], np.uint32)
        result = compute(
            device, lambda x, n: dml.reverse_subsequences(x, n, axis=3), array, lengths)
        assert np.array_equal(result[0, 0], [[2.0, 1.0, 3.0, 4.0],
                                             [8.0, 7.0, 6.0, 5.0]])

    def test_resample_repeats_the_nearest_neighbour(self, device):
        # Pixel offsets of zero put output pixel i at input i / 2, and a tie
        # broken downwards makes that exactly numpy's repeat. The defaults are
        # the half-pixel grid, which samples 0.25, 0.75, 1.25, 1.75 instead.
        result = compute(
            device,
            lambda x: dml.resample(
                x, output_sizes=[1, 1, 4, 4],
                mode=dml.InterpolationMode.NEAREST_NEIGHBOR,
                rounding_direction=dml.AxisDirection.DECREASING,
                input_pixel_offsets=[0.0] * 4, output_pixel_offsets=[0.0] * 4),
            A)
        assert result.shape == (1, 1, 4, 4)
        assert np.array_equal(result, np.repeat(np.repeat(A, 2, axis=2), 2, axis=3))


class TestFill:
    """The two operators that produce a tensor from nothing but a description."""

    def test_constant(self, device):
        graph = dml.Graph(device)
        filled = dml.fill_value_constant(graph, sizes=[1, 1, 2, 2], value=2.5)
        result, = graph.compile([filled])()
        assert result.dtype == np.float32
        assert np.all(result == 2.5)

    def test_constant_takes_the_type_it_is_given(self, device):
        graph = dml.Graph(device)
        filled = dml.fill_value_constant(graph, sizes=[1, 1, 2, 2], value=-3,
                                         dtype=np.int32)
        result, = graph.compile([filled])()
        assert result.dtype == np.int32
        assert np.all(result == -3)

    def test_constant_refuses_a_value_the_type_cannot_hold(self, device):
        graph = dml.Graph(device)
        with pytest.raises(ValueError, match="value=300"):
            dml.fill_value_constant(graph, sizes=[1, 1, 2, 2], value=300,
                                    dtype=np.uint8)

    def test_sequence_counts_in_memory_order(self, device):
        graph = dml.Graph(device)
        # The identity is not decoration: DirectML's graph compiler faults on a
        # graph whose output is the sequence node itself, as the wrapper says.
        filled = dml.identity(dml.fill_value_sequence(
            graph, sizes=[1, 1, 2, 3], value_start=1.0, value_delta=0.5))
        result, = graph.compile([filled])()
        assert np.allclose(result[0, 0], [[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])


class TestReductions:
    @pytest.mark.parametrize("function, reference", [
        (dml.ReduceFunction.SUM, np.sum),
        (dml.ReduceFunction.AVERAGE, np.mean),
        (dml.ReduceFunction.MAX, np.max),
        (dml.ReduceFunction.MIN, np.min),
        (dml.ReduceFunction.MULTIPLY, np.prod),
        (dml.ReduceFunction.SUM_SQUARE, lambda a, axis: np.sum(a * a, axis=axis)),
        (dml.ReduceFunction.L1, lambda a, axis: np.sum(np.abs(a), axis=axis)),
        (dml.ReduceFunction.L2, lambda a, axis: np.sqrt(np.sum(a * a, axis=axis))),
    ])
    def test_reduce_over_one_axis(self, device, function, reference):
        result = compute(device, lambda x: dml.reduce(x, function=function, axes=[3]), A)
        assert result.shape == (1, 1, 2, 1)
        assert np.allclose(result.ravel(), reference(A, axis=3).ravel(), rtol=1e-4)

    def test_reduce_defaults_to_every_axis(self, device):
        result = compute(device, lambda x: dml.reduce(x, function=dml.ReduceFunction.SUM), A)
        assert result.shape == (1, 1, 1, 1)
        assert np.allclose(result.ravel(), [A.sum()])

    def test_argmax_writes_an_index(self, device):
        result = compute(
            device,
            lambda x: dml.reduce(x, function=dml.ReduceFunction.ARGMAX, axes=[3],
                                 output_dtype=np.uint32),
            ROW)
        assert result.dtype == np.uint32
        assert np.array_equal(result.ravel(), [1])

    def test_cumulative_summation(self, device):
        result = compute(device, lambda x: dml.cumulative_summation(x, axis=3), ROW)
        assert np.allclose(result.ravel(), np.cumsum(ROW.ravel()))

    def test_cumulative_summation_can_run_backwards_and_exclusively(self, device):
        result = compute(
            device,
            lambda x: dml.cumulative_summation(
                x, axis=3, axis_direction=dml.AxisDirection.DECREASING,
                has_exclusive_sum=True),
            ROW)
        values = ROW.ravel()
        assert np.allclose(result.ravel(), [values[i + 1:].sum() for i in range(4)])

    def test_cumulative_product(self, device):
        result = compute(device, lambda x: dml.cumulative_product(x, axis=3), ROW)
        assert np.allclose(result.ravel(), np.cumprod(ROW.ravel()))

    def test_non_zero_coordinates(self, device):
        array = shaped([[0.0, 2.0], [3.0, 0.0]])
        graph = dml.Graph(device)
        x = graph.input(array.shape, array.dtype)
        outputs = dml.non_zero_coordinates(x)
        count, coordinates = graph.compile(list(outputs))({x: array})
        assert count.ravel()[0] == 2
        assert np.array_equal(coordinates[:2], [[0, 0, 0, 1], [0, 0, 1, 0]])


class TestQuantization:
    def test_round_trip(self, device):
        array = shaped([0.0, 1.0, 2.0, 3.0])
        scale = shaped([0.5, 0.5, 0.5, 0.5])
        zero_point = shaped([2, 2, 2, 2], np.uint8)

        graph = dml.Graph(device)
        x = graph.input(array.shape, array.dtype)
        s = graph.input(scale.shape, scale.dtype)
        z = graph.input(zero_point.shape, zero_point.dtype)
        quantized = dml.quantize_linear(x, s, z)
        op = graph.compile([quantized, dml.dequantize_linear(quantized, s, z)])
        integers, floats = op({x: array, s: scale, z: zero_point})

        assert integers.dtype == np.uint8
        assert np.array_equal(integers.ravel(), [2, 4, 6, 8])
        assert np.allclose(floats.ravel(), array.ravel())


class TestRandomGenerator:
    def test_values_and_state(self, device):
        state = np.zeros((1, 1, 1, 6), np.uint32)
        graph = dml.Graph(device)
        x = graph.input(state.shape, state.dtype)
        outputs = dml.random_generator(x, output_sizes=[1, 1, 1, 4])
        values, next_state = graph.compile(list(outputs))({x: state})

        assert values.dtype == np.uint32
        assert values.shape == (1, 1, 1, 4)
        assert len(set(values.ravel().tolist())) > 1
        assert not np.array_equal(next_state, state)

    def test_state_is_optional(self, device):
        state = np.zeros((1, 1, 1, 6), np.uint32)
        graph = dml.Graph(device)
        x = graph.input(state.shape, state.dtype)
        outputs = dml.random_generator(x, output_sizes=[1, 1, 1, 4], output_state=False)
        assert outputs.state is None
        values, = graph.compile([outputs.values])({x: state})
        assert values.shape == (1, 1, 1, 4)


class TestRoiAlign:
    def test_a_flat_region_pools_to_its_value(self, device):
        # Whatever the sampling grid, the average of a constant is that
        # constant: this checks the wiring, not DirectML's interpolation.
        array = np.full((1, 1, 4, 4), 3.0, np.float32)
        roi = shaped([[0.0, 0.0, 4.0, 4.0]])
        batch_indices = shaped([0], np.uint32)

        result = compute(
            device,
            lambda x, r, b: dml.roi_align(
                x, r, b,
                reduction_function=dml.ReduceFunction.AVERAGE,
                interpolation_mode=dml.InterpolationMode.LINEAR,
                spatial_scale_x=1.0, spatial_scale_y=1.0,
                input_pixel_offset=0.5, output_pixel_offset=-0.5,
                out_of_bounds_input_value=0.0,
                minimum_samples_per_output=1, maximum_samples_per_output=1,
                align_regions_to_corners=False,
                output_height=2, output_width=2),
            array, roi, batch_indices)

        assert result.shape == (1, 1, 2, 2)
        assert np.allclose(result, 3.0)
