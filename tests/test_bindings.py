"""The rules that make dict-shaped binding safe.

These are the contracts most easily broken without noticing: the dtype cast
table, the errors that replace positional binding's silent garbage, the byte
padding at upload, and the initialize/dispatch split. The graphs are tiny; a
dispatch on any device suffices.
"""

import numpy as np
import pytest

import directml as dml


@pytest.fixture(scope="module")
def device():
    return dml.Device()


def identity_op(device, sizes, dtype=np.float32):
    graph = dml.Graph(device)
    x = graph.input(sizes, dtype)
    op = graph.compile([dml.activation_identity(x)])
    return x, op


class TestCastTable:
    """The table from docs/api-design.md section 4.1, row by row: within a kind
    or NumPy-safe converts silently, everything else must be an explicit astype."""

    def dispatch(self, device, array, dtype=dml.TensorDataType.FLOAT32):
        x, op = identity_op(device, [1, 1, 2, 2], dtype)
        return op({x: array})[0]

    def test_float64_narrows_to_float32(self, device):
        result = self.dispatch(device, np.full((2, 2), 1.5, np.float64))
        assert result.dtype == np.float32
        assert np.all(result == 1.5)

    def test_float32_narrows_to_float16(self, device):
        result = self.dispatch(device, np.full((2, 2), 0.5, np.float32),
                               np.float16)
        assert result.dtype == np.float16
        assert np.all(result == 0.5)

    def test_uint8_widens_to_float32(self, device):
        result = self.dispatch(device, np.arange(4, dtype=np.uint8).reshape(2, 2))
        assert result.dtype == np.float32
        assert np.all(result.ravel() == [0, 1, 2, 3])

    def test_int32_to_float32_is_refused(self, device):
        with pytest.raises(ValueError, match="astype"):
            self.dispatch(device, np.zeros((2, 2), np.int32))

    def test_float32_to_int32_is_refused(self, device):
        # Neither ACTIVATION_IDENTITY nor the fused-activation form of add
        # supports integers, so build the tensor into a subtract instead.
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2], np.int32)
        op = graph.compile([dml.subtract(x, x)])
        with pytest.raises(ValueError, match="astype"):
            op({x: np.zeros((2, 2), np.float32)})


class TestBindingErrors:
    def test_missing_input_is_named(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2])
        b = graph.input([1, 1, 2, 2])
        op = graph.compile([dml.add(a, b)])
        with pytest.raises(ValueError, match="missing input 1"):
            op({a: np.zeros((2, 2), np.float32)})

    def test_non_input_expression_is_refused(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        y = dml.activation_identity(x)
        op = graph.compile([y])
        with pytest.raises(ValueError, match="not an input"):
            op({y: np.zeros((2, 2), np.float32)})

    def test_owned_input_is_refused_at_dispatch(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(x, w)])
        op.initialize({w: np.zeros((2, 2), np.float32)})
        with pytest.raises(ValueError, match="initialize"):
            op({x: np.zeros((2, 2), np.float32), w: np.zeros((2, 2), np.float32)})

    def test_plain_input_is_refused_at_initialize(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(x, w)])
        with pytest.raises(ValueError, match="dispatch"):
            op.initialize({w: np.zeros((2, 2), np.float32),
                           x: np.zeros((2, 2), np.float32)})

    def test_wrong_element_count_is_refused(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        with pytest.raises(ValueError, match="does not fill"):
            op({x: np.zeros((2, 3), np.float32)})


class TestOperators:
    """Arithmetic on Expression comes from DirectMLX's C++ overloads, with the
    deviations the README documents: % is floored like Python's, a float
    numerator scales the reciprocal, and there are no in-place forms."""

    def compute(self, device, build, *arrays):
        graph = dml.Graph(device)
        inputs = [graph.input(a.shape, a.dtype) for a in arrays]
        op = graph.compile([build(*inputs)])
        return op(dict(zip(inputs, arrays)))[0]

    def test_expression_operands(self, device):
        a = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
        b = np.full((1, 1, 2, 2), 2.0, np.float32)
        result = self.compute(device, lambda x, y: x * y - x / y, a, b)
        assert np.allclose(result, a * b - a / b)

    def test_float_operands_ride_the_scale_bias(self, device):
        a = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        result = self.compute(device, lambda x: 1.0 - x * 0.5, a)
        assert np.allclose(result, 1.0 - a * 0.5)

    def test_negation(self, device):
        a = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        assert np.allclose(self.compute(device, lambda x: -x, a), -a)

    def test_float_numerator_scales_the_reciprocal(self, device):
        # DirectMLX's own operator/ hands the scale to Recip, and an
        # elementwise scale-bias applies to the input: 1/(2x), not 2/x.
        a = np.array([1, 2, 4, 8], np.float32).reshape(1, 1, 2, 2)
        result = self.compute(device, lambda x: 2.0 / x, a)
        assert np.allclose(result, 2.0 / a)

    def test_modulus_is_floored_like_pythons(self, device):
        a = np.array([-7, -1, 1, 7], np.int32).reshape(1, 1, 2, 2)
        b = np.full((1, 1, 2, 2), 5, np.int32)
        result = self.compute(device, lambda x, y: x % y, a, b)
        assert np.array_equal(result, a % b)

    def test_augmented_assignment_rebinds_the_name(self, device):
        # Deliberately no __iadd__: it would mutate the node behind every
        # reference to it, changing the identity that dict binding matches
        # on. The x = x + y fallback rebinds one name and leaves the input
        # reachable through its alias.
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        alias = x
        x += 1.0
        assert x is not alias
        op = graph.compile([x])
        result, = op({alias: np.zeros((2, 2), np.float32)})
        assert np.all(result == 1.0)


class TestByteSizes:
    def test_short_packed_array_is_padded(self, device):
        # Three float16s are 6 bytes; DirectML rounds the tensor up to 8. The
        # upload must pad rather than read past the array's end.
        x, op = identity_op(device, [1, 1, 1, 3], np.float16)
        result, = op({x: np.array([1, 2, 3], np.float16)})
        assert result.tolist() == [[[[1, 2, 3]]]]

    def test_flat_array_fills_the_shape(self, device):
        # The library knows the shape; a flat array of the right length is fine.
        x, op = identity_op(device, [1, 1, 2, 2])
        result, = op({x: np.arange(4, dtype=np.float32)})
        assert result.shape == (1, 1, 2, 2)


class TestExecution:
    def test_dispatch_without_initialize_names_the_owned_inputs(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(x, w)])
        with pytest.raises(ValueError, match=r"initialize\(\).*input 1"):
            op({x: np.zeros((2, 2), np.float32)})

    def test_no_owned_inputs_skips_initialize(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        result, = op({x: np.full((2, 2), 3.0, np.float32)})
        assert np.all(result == 3.0)

    def test_initialized_weights_survive_the_graph(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 3])
        w = graph.input([1, 1, 3, 2], owned=True)
        op = graph.compile([dml.gemm(a, w)])

        weights = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
        op.initialize({w: weights})
        del graph

        activations = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
        result, = op({a: activations})
        assert np.allclose(result[0, 0], activations[0, 0] @ weights[0, 0])

    def test_reinitializing_replaces_the_weights(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(x, w)])

        zeros = np.zeros((2, 2), np.float32)
        op.initialize({w: np.full((2, 2), 1.0, np.float32)})
        assert np.all(op({x: zeros})[0] == 1.0)
        op.initialize({w: np.full((2, 2), 2.0, np.float32)})
        assert np.all(op({x: zeros})[0] == 2.0)

    def test_outputs_come_back_shaped_and_typed(self, device):
        x, op = identity_op(device, [1, 2, 3, 4])
        result, = op({x: np.zeros((1, 2, 3, 4), np.float32)})
        assert result.shape == (1, 2, 3, 4)
        assert result.dtype == np.float32


class TestConstants:
    """graph.constant() records the data with the declaration; compile()
    uploads it and lets it go."""

    def test_compile_initializes_a_graph_of_constants(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.constant(np.full((1, 1, 2, 2), 2.0, np.float32))
        op = graph.compile([dml.multiply(x, w)])
        assert op.initialized
        result, = op({x: np.full((2, 2), 3.0, np.float32)})
        assert np.all(result == 6.0)

    def test_constant_takes_the_arrays_dtype_and_a_view_shape(self, device):
        graph = dml.Graph(device)
        w = graph.constant(np.arange(4, dtype=np.int32), sizes=[1, 1, 2, 2])
        assert w.dtype == np.int32
        assert w.shape == (1, 1, 2, 2)

    def test_constant_converts_at_upload(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2], np.float16)
        w = graph.constant(np.full((1, 1, 2, 2), 0.5, np.float64), np.float16)
        op = graph.compile([dml.add(x, w)])
        result, = op({x: np.zeros((2, 2), np.float16)})
        assert result.dtype == np.float16
        assert np.all(result == 0.5)

    def test_constant_refuses_an_unsafe_cast_at_declaration(self, device):
        graph = dml.Graph(device)
        with pytest.raises(ValueError, match="astype"):
            graph.constant(np.zeros((2, 2), np.int32), np.float32)

    def test_constant_refuses_a_shape_it_does_not_fill(self, device):
        graph = dml.Graph(device)
        with pytest.raises(ValueError, match="does not fill"):
            graph.constant(np.zeros((2, 2), np.float32), sizes=[1, 1, 2, 3])

    def test_constants_wait_for_the_other_owned_inputs(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        c = graph.constant(np.full((1, 1, 2, 2), 1.0, np.float32))
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(dml.add(x, c), w)])
        assert not op.initialized
        # The constant need not be named again; the plain owned input must be.
        with pytest.raises(ValueError, match="missing input 2"):
            op.initialize({})
        op.initialize({w: np.full((2, 2), 2.0, np.float32)})
        result, = op({x: np.zeros((2, 2), np.float32)})
        assert np.all(result == 3.0)

    def test_reinitializing_needs_the_constants_again(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        c = graph.constant(np.full((1, 1, 2, 2), 1.0, np.float32))
        op = graph.compile([dml.add(x, c)])
        with pytest.raises(ValueError, match="missing input 1"):
            op.initialize({})
        op.initialize({c: np.full((2, 2), 5.0, np.float32)})
        result, = op({x: np.zeros((2, 2), np.float32)})
        assert np.all(result == 5.0)

    def test_graph_drops_the_arrays_at_compile(self, device):
        graph = dml.Graph(device)
        w = graph.constant(np.zeros((1, 1, 2, 2), np.float32))
        assert graph._constants
        graph.compile([dml.activation_identity(w)])
        assert "_constants" not in graph.__dict__


class TestNames:
    def test_named_inputs_bind_by_name(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2], name="a")
        b = graph.input([1, 1, 2, 2], name="b")
        op = graph.compile([dml.subtract(a, b)])
        result, = op({"a": np.full((2, 2), 3.0, np.float32),
                      b: np.full((2, 2), 1.0, np.float32)})
        assert np.all(result == 2.0)

    def test_unknown_name_is_refused(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2], name="a")
        op = graph.compile([dml.activation_identity(a)])
        with pytest.raises(ValueError, match="no input named 'b'"):
            op({"b": np.zeros((2, 2), np.float32)})

    def test_duplicate_name_is_refused(self, device):
        graph = dml.Graph(device)
        graph.input([1, 1, 2, 2], name="a")
        with pytest.raises(ValueError, match="already has an input named 'a'"):
            graph.constant(np.zeros((1, 1, 2, 2), np.float32), name="a")

    def test_binding_twice_is_refused(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2], name="a")
        op = graph.compile([dml.activation_identity(a)])
        with pytest.raises(ValueError, match="'a'.*bound twice"):
            op({"a": np.zeros((2, 2), np.float32), a: np.zeros((2, 2), np.float32)})

    def test_errors_name_the_input(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2], name="latent")
        op = graph.compile([dml.activation_identity(x)])
        with pytest.raises(ValueError, match="input 0 'latent'"):
            op({})


class TestBuffers:
    """A Buffer is a tensor on the GPU: uploaded once, bound in place, and
    handed out by dispatch(readback=False) so the next graph can bind it."""

    def test_round_trip(self, device):
        array = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
        buffer = dml.Buffer(device, array)
        assert buffer.shape == (1, 1, 2, 3)
        assert buffer.dtype == np.float32
        assert buffer.nbytes == 24
        assert np.array_equal(buffer.numpy(), array)

    def test_upload_converts_under_the_cast_table(self, device):
        buffer = dml.Buffer(device, np.full((2, 2), 0.5, np.float64), np.float16)
        assert buffer.dtype == np.float16
        assert np.all(buffer.numpy() == 0.5)
        with pytest.raises(ValueError, match="astype"):
            dml.Buffer(device, np.zeros((2, 2), np.int32), np.float32)

    def test_buffer_binds_as_an_input(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        buffer = dml.Buffer(device, np.full((1, 1, 2, 2), 4.0, np.float32))
        result, = op({x: buffer})
        assert np.all(result == 4.0)

    def test_buffer_dtype_must_match(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        buffer = dml.Buffer(device, np.zeros((1, 1, 2, 2), np.float16))
        with pytest.raises(ValueError, match="not converted"):
            op({x: buffer})

    def test_buffer_must_fill_the_tensor(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        buffer = dml.Buffer(device, np.zeros((1, 1, 2, 3), np.float32))
        with pytest.raises(ValueError, match="does not fill"):
            op({x: buffer})

    def test_outputs_stay_on_the_gpu_and_feed_the_next_graph(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        first = graph.compile([x * 2.0, x + 1.0])

        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2])
        b = graph.input([1, 1, 2, 2])
        second = graph.compile([dml.multiply(a, b)])

        values = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        doubled, incremented = first({x: values}, readback=False)
        assert isinstance(doubled, dml.Buffer)
        assert doubled.shape == (1, 1, 2, 2)
        result, = second({a: doubled, b: incremented})
        assert np.array_equal(result, values * 2 * (values + 1))
        assert np.array_equal(doubled.numpy(), values * 2)

    def test_buffer_initializes_an_owned_input(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.input([1, 1, 2, 2], owned=True)
        op = graph.compile([dml.add(x, w)])
        op.initialize({w: dml.Buffer(device, np.full((1, 1, 2, 2), 7.0, np.float32))})
        result, = op({x: np.zeros((2, 2), np.float32)})
        assert np.all(result == 7.0)

    def test_buffer_as_a_constant(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        w = graph.constant(dml.Buffer(device, np.full((1, 1, 2, 2), 7.0, np.float32)))
        op = graph.compile([dml.add(x, w)])
        result, = op({x: np.zeros((2, 2), np.float32)})
        assert np.all(result == 7.0)

    def test_buffer_from_another_device_is_refused(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        other = dml.Device()
        buffer = dml.Buffer(other, np.zeros((1, 1, 2, 2), np.float32))
        with pytest.raises(ValueError, match="different device"):
            op({x: buffer})

    def test_buffer_outlives_its_graph_and_operator(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        buffer, = op({x: np.full((2, 2), 9.0, np.float32)}, readback=False)
        del op, x
        assert np.all(buffer.numpy() == 9.0)


class TestBroadcast:
    def test_axes_of_one_get_a_zero_stride(self, device):
        graph = dml.Graph(device)
        view = dml.broadcast(graph.input([1, 3, 1, 1]), [2, 3, 4, 5])
        assert view.shape == (2, 3, 4, 5)
        assert view.strides == (0, 1, 0, 0)

    def test_leading_axes_are_added(self, device):
        graph = dml.Graph(device)
        view = dml.broadcast(graph.input([3, 4]), [2, 3, 4])
        assert view.strides == (0, 4, 1)

    def test_same_shape_is_the_same_expression(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 1, 2, 2])
        assert dml.broadcast(x, [1, 1, 2, 2]) == x

    def test_existing_strides_are_kept(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 2, 3, 4])
        transposed = dml.reinterpret(x, [1, 2, 4, 3], [24, 12, 1, 4])
        view = dml.broadcast(transposed, [5, 2, 4, 3])
        assert view.strides == (0, 12, 1, 4)

    def test_mismatch_is_refused(self, device):
        graph = dml.Graph(device)
        x = graph.input([1, 3, 2, 2])
        with pytest.raises(ValueError, match="cannot broadcast"):
            dml.broadcast(x, [1, 3, 4, 4])
        with pytest.raises(ValueError, match="fewer axes"):
            dml.broadcast(x, [3, 2, 2])

    def test_broadcast_computes(self, device):
        graph = dml.Graph(device)
        image = graph.input([1, 3, 2, 2])
        bias = graph.constant(np.array([1, 2, 3], np.float32), sizes=[1, 3, 1, 1])
        op = graph.compile([image + dml.broadcast(bias, image.shape)])
        result, = op({image: np.zeros((1, 3, 2, 2), np.float32)})
        assert np.array_equal(result[0, :, 1, 1], [1, 2, 3])


class TestElementwiseChecks:
    """A mismatched elementwise pair is refused where it is written, naming
    both operands, rather than at compile as a bare E_INVALIDARG."""

    def test_shape_mismatch_names_the_operands(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2])
        b = graph.input([1, 1, 2, 3])
        with pytest.raises(ValueError, match=r"\[1, 1, 2, 2\].*\[1, 1, 2, 3\].*broadcast"):
            a + b
        with pytest.raises(ValueError, match="shapes differ"):
            dml.add(a, b)

    def test_dtype_mismatch_is_refused(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2], np.float32)
        b = graph.input([1, 1, 2, 2], np.float16)
        with pytest.raises(ValueError, match="element types differ"):
            a * b

    def test_float_operands_are_not_checked(self, device):
        graph = dml.Graph(device)
        a = graph.input([1, 1, 2, 2])
        assert (a * 2.0).shape == (1, 1, 2, 2)


class TestTensorDesc:
    def test_accepts_numpy_dtypes(self):
        desc = dml.TensorDesc(np.float16, [2, 3])
        assert desc.data_type == dml.TensorDataType.FLOAT16
        assert desc.total_tensor_size_in_bytes == 12

    def test_is_the_class_the_library_hands_back(self, device):
        # One class, not a Python subclass over a C++ one: what an Expression
        # reports is a dml.TensorDesc too.
        graph = dml.Graph(device)
        assert isinstance(graph.input([1, 1, 2, 2]).desc, dml.TensorDesc)
