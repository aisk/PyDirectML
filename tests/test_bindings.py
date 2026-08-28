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


def make_input(graph, index, sizes, dtype=dml.TensorDataType.FLOAT32, owned=False):
    flags = dml.TensorFlags.OWNED_BY_DML if owned else dml.TensorFlags.NONE
    return dml.input_tensor(graph, index, dml.TensorDesc(dtype, flags, sizes))


def identity_op(device, sizes, dtype=dml.TensorDataType.FLOAT32):
    graph = dml.GraphBuilder(device)
    x = make_input(graph, 0, sizes, dtype)
    op = graph.build(dml.ExecutionFlags.NONE, [dml.activation_identity(x)])
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
                               dml.TensorDataType.FLOAT16)
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
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2], dml.TensorDataType.INT32)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.subtract(x, x)])
        with pytest.raises(ValueError, match="astype"):
            op({x: np.zeros((2, 2), np.float32)})


class TestBindingErrors:
    def test_missing_input_is_named(self, device):
        graph = dml.GraphBuilder(device)
        a = make_input(graph, 0, [1, 1, 2, 2])
        b = make_input(graph, 1, [1, 1, 2, 2])
        op = graph.build(dml.ExecutionFlags.NONE, [dml.add(a, b)])
        with pytest.raises(ValueError, match="missing input 1"):
            op({a: np.zeros((2, 2), np.float32)})

    def test_non_input_expression_is_refused(self, device):
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2])
        y = dml.activation_identity(x)
        op = graph.build(dml.ExecutionFlags.NONE, [y])
        with pytest.raises(ValueError, match="not an input"):
            op({y: np.zeros((2, 2), np.float32)})

    def test_owned_input_is_refused_at_dispatch(self, device):
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2])
        w = make_input(graph, 1, [1, 1, 2, 2], owned=True)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.add(x, w)])
        op.initialize({w: np.zeros((2, 2), np.float32)})
        with pytest.raises(ValueError, match="initialize"):
            op({x: np.zeros((2, 2), np.float32), w: np.zeros((2, 2), np.float32)})

    def test_plain_input_is_refused_at_initialize(self, device):
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2])
        w = make_input(graph, 1, [1, 1, 2, 2], owned=True)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.add(x, w)])
        with pytest.raises(ValueError, match="dispatch"):
            op.initialize({w: np.zeros((2, 2), np.float32),
                           x: np.zeros((2, 2), np.float32)})

    def test_wrong_element_count_is_refused(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        with pytest.raises(ValueError, match="does not fill"):
            op({x: np.zeros((2, 3), np.float32)})


class TestByteSizes:
    def test_short_packed_array_is_padded(self, device):
        # Three float16s are 6 bytes; DirectML rounds the tensor up to 8. The
        # upload must pad rather than read past the array's end.
        x, op = identity_op(device, [1, 1, 1, 3], dml.TensorDataType.FLOAT16)
        result, = op({x: np.array([1, 2, 3], np.float16)})
        assert result.tolist() == [[[[1, 2, 3]]]]

    def test_flat_array_fills_the_shape(self, device):
        # The library knows the shape; a flat array of the right length is fine.
        x, op = identity_op(device, [1, 1, 2, 2])
        result, = op({x: np.arange(4, dtype=np.float32)})
        assert result.shape == (1, 1, 2, 2)


class TestExecution:
    def test_dispatch_without_initialize_names_the_owned_inputs(self, device):
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2])
        w = make_input(graph, 1, [1, 1, 2, 2], owned=True)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.add(x, w)])
        with pytest.raises(ValueError, match=r"initialize\(\).*input 1"):
            op({x: np.zeros((2, 2), np.float32)})

    def test_no_owned_inputs_skips_initialize(self, device):
        x, op = identity_op(device, [1, 1, 2, 2])
        result, = op({x: np.full((2, 2), 3.0, np.float32)})
        assert np.all(result == 3.0)

    def test_initialized_weights_survive_the_graph(self, device):
        graph = dml.GraphBuilder(device)
        a = make_input(graph, 0, [1, 1, 2, 3])
        w = make_input(graph, 1, [1, 1, 3, 2], owned=True)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.gemm(a, w)])

        weights = np.arange(6, dtype=np.float32).reshape(1, 1, 3, 2)
        op.initialize({w: weights})
        del graph

        activations = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
        result, = op({a: activations})
        assert np.allclose(result[0, 0], activations[0, 0] @ weights[0, 0])

    def test_reinitializing_replaces_the_weights(self, device):
        graph = dml.GraphBuilder(device)
        x = make_input(graph, 0, [1, 1, 2, 2])
        w = make_input(graph, 1, [1, 1, 2, 2], owned=True)
        op = graph.build(dml.ExecutionFlags.NONE, [dml.add(x, w)])

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
