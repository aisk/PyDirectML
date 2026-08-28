"""Multiply two matrices on the GPU with DirectML.

DirectML's GEMM operator works on 4-D tensors, so an (M, K) matrix is fed in as
a [1, 1, M, K] tensor and the [1, 1, M, N] result is reshaped back to 2-D.
"""

import numpy as np

import directml as dml


def matmul(a, b):
    """Return the matrix product ``a @ b`` of two 2-D arrays, computed on the GPU."""
    device = dml.Device()
    graph = dml.Graph(device)

    lhs = graph.input([1, 1, *a.shape])
    rhs = graph.input([1, 1, *b.shape])
    op = graph.compile([dml.gemm(lhs, rhs)])

    result, = op({lhs: a, rhs: b})
    return result.reshape(a.shape[0], b.shape[1])


def main():
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    b = np.arange(1, 21, dtype=np.float32).reshape(4, 5)

    product = matmul(a, b)

    print(f"A ({a.shape[0]}x{a.shape[1]}):", a, sep="\n")
    print(f"\nB ({b.shape[0]}x{b.shape[1]}):", b, sep="\n")
    print(f"\nA @ B ({product.shape[0]}x{product.shape[1]}):", product, sep="\n")

    if np.allclose(product, a @ b):
        print("\nMatches NumPy.")
    else:
        print("\nDoes not match NumPy:", a @ b, sep="\n")


if __name__ == "__main__":
    main()
