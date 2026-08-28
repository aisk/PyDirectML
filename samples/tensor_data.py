"""np.load for the samples' checked-in .npy files, tolerating their headers.

The tensor data next to these samples was written by a converter whose header
formatting NumPy 2.0's stricter parser rejects (no space after the colons).
The bytes are fine; only the header dict needs a lenient read.
"""

import ast

import numpy as np


def load(path):
    try:
        return np.load(path)
    except ValueError:
        with open(path, "rb") as file:
            magic = file.read(8)  # \x93NUMPY plus a two-byte version
            if magic[:6] != b"\x93NUMPY":
                raise
            length_bytes = 4 if magic[6] >= 2 else 2
            length = int.from_bytes(file.read(length_bytes), "little")
            # The padding here is a newline followed by spaces, which even
            # ast.literal_eval trips over; strip it before parsing.
            header = ast.literal_eval(file.read(length).decode("latin1").strip())
            data = np.fromfile(file, np.dtype(header["descr"]))
        return data.reshape(header["shape"],
                            order="F" if header["fortran_order"] else "C")
