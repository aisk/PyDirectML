"""Run the same graph at float32 and at float16, and show what Binding refuses.

A tensor's ``TensorDataType`` decides the dtype on both ends: ``Binding`` converts
what you hand it to that type, and results come back as that type. Half precision
halves the memory a set of weights takes and the bytes moved to reach them, which
is what makes it interesting for anything model-sized.

Conversions that stay within a dtype kind, or that NumPy calls safe, happen
silently. Anything else has to be spelled out, because reinterpreting an integer
array as floating point is a wrong answer rather than an error.
"""

import numpy as np

import directml as dml


def gemm(device, data_type, a, b):
    """Multiply two matrices with every tensor declared as ``data_type``."""
    graph = dml.GraphBuilder(device)
    lhs = dml.input_tensor(graph, 0, dml.TensorDesc(data_type, [1, 1, *a.shape]))
    rhs = dml.input_tensor(graph, 1, dml.TensorDesc(data_type, [1, 1, *b.shape]))
    product = dml.gemm(lhs, rhs)

    op = graph.build(dml.ExecutionFlags.NONE, [product])
    output, = device.compute(op, [dml.Binding(lhs, a), dml.Binding(rhs, b)], [product])

    # No dtype is named here: the buffer NumPy reads carries the tensor's own.
    return np.asarray(output).reshape(a.shape[0], b.shape[1])


def main():
    device = dml.Device(use_gpu=True)
    a = (np.arange(1, 13).reshape(3, 4) / 7.0)
    b = (np.arange(1, 21).reshape(4, 5) / 7.0)
    expected = a @ b

    print("Same graph, two tensor data types")
    for data_type in (dml.TensorDataType.FLOAT32, dml.TensorDataType.FLOAT16):
        product = gemm(device, data_type, a, b)
        error = np.abs(product.astype(np.float64) - expected).max()
        print(f"  {str(data_type):<32} -> {product.dtype}, "
              f"{product.nbytes} bytes, max error {error:.2e}")

    print("\nWhat Binding accepts")
    graph = dml.GraphBuilder(device)
    tensor = dml.input_tensor(graph, 0, dml.TensorDesc(dml.TensorDataType.FLOAT32, [1, 1, 2, 2]))

    cases = [
        ("float64 array", np.zeros((2, 2), np.float64)),
        ("uint8 array", np.zeros((2, 2), np.uint8)),
        ("int32 array", np.zeros((2, 2), np.int32)),
        ("int32 array, converted", np.zeros((2, 2), np.int32).astype(np.float32)),
        ("float32 array of the wrong size", np.zeros((2, 3), np.float32)),
    ]
    for label, array in cases:
        try:
            dml.Binding(tensor, array)
            print(f"  {label:<32} accepted")
        except ValueError as error:
            print(f"  {label:<32} {error}")


if __name__ == "__main__":
    main()
