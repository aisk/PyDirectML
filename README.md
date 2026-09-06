# Python Binding for DirectML

> Forked from the `Python/` directory of [microsoft/DirectML](https://github.com/microsoft/DirectML), which is no longer maintained. `third_party/DirectMLX.h` comes from that repository's `Libraries/` directory. Licensed under MIT, see [LICENSE](./LICENSE).

`directml` is a small Python binding for DirectML: DirectMLX's operators as Python functions, a device and resource layer that stays out of the way, and NumPy arrays in and out. It exists to write DirectML models in Python without a framework in between; `samples/sdxl/` runs Stable Diffusion XL on it.

## Usage

```python
import numpy as np
import directml as dml

device = dml.Device()
graph = dml.Graph(device)

x = graph.input([1, 1, 28, 28], name="image")            # float32 is the default
w = graph.constant(np.load("w.npy"))                     # a weight, uploaded at compile
b = graph.constant(np.load("b.npy"), sizes=[1, 8, 1, 1])
conv = dml.convolution(x, w, strides=[1, 1],
                       start_padding=[2, 2], end_padding=[2, 2])
conv = conv + dml.broadcast(b, conv.shape)               # a [1, 8, 1, 1] bias, per channel
probs = dml.activation_softmax(conv, axes=[1])

op = graph.compile([probs])                              # initialized: every weight is a constant
result, = op({"image": image})                           # a shaped, typed ndarray
```

An input declared with `owned=True` instead of `constant()` is bound once at `op.initialize({w: array})` and lives on the GPU from then on. `op(inputs, readback=False)` leaves the outputs on the GPU as `dml.Buffer`s, which the next graph binds in place.

DirectML's tensors are 4-D (up to 8-D for some operators), and the bindings expose that as it is: a matrix is fed to `gemm` as `[1, 1, M, K]`, and a `[C]` bias is viewed as `[1, C, 1, 1]` before it is broadcast. `samples/matmul.py` shows the reshape on the way in and out.

The API and the reasoning behind it are in [docs/api-design.md](./docs/api-design.md).

## Differences from upstream

- The module is imported as **`directml`**. `DirectML.h` and `DirectML.lib` come from the Windows SDK and the extension loads the `DirectML.dll` that ships with Windows; nothing is bundled.
- Inputs are bound as a **dict** from `Expression` (or the `name=` it was given) to an array or `Buffer`, not a list matched by position. `op.initialize({...})` takes the tensors DirectML owns, `op({...})` the rest; a missing, extra, misplaced or mistyped input is a `ValueError` that names it. The library keeps no CPU copy of any weight.
- **`graph.constant(array)`** declares an owned input and records its data; `compile()` uploads the constants and initializes the operator when they are its only owned inputs.
- **`dml.Buffer`** is a tensor that stays on the GPU: `op(inputs, readback=False)` returns one per output, a binding dict accepts one in place of an array, `buffer.numpy()` reads it back.
- **`dml.broadcast(x, shape)`** views a tensor through a larger shape as a zero-stride `reinterpret`. Nothing broadcasts implicitly, and a mismatched elementwise pair is refused where it is written, naming both operands.
- A tensor's data type is honored on both ends and every API that takes one accepts numpy dtypes. Arrays are converted to the tensor's type when the conversion stays within a dtype kind or NumPy calls it safe; anything else must be an explicit `astype()`. Results come back as arrays of the tensor's type and shape.
- `initialize` and `dispatch` live on the `CompiledOperator`, which owns its persistent resource, so weights are uploaded once and stay on the GPU. Upstream re-initialized on every `compute`.
- **Every operator DirectMLX wraps for inference is bound**: the elementwise unaries with the `scale_bias=` DirectML folds into their input, the binaries and comparisons, the activations, `reduce`, `resample`, the gathers and scatters, `top_k`, `roi_align`, the fills, and `multihead_attention`, which DirectMLX itself does not wrap. The training side is bound too: the `*Grad` operators DirectML implements, `batch_normalization_training`, the two integer convolutions and the feature-level 6.3 blocked `Dequantize`. Nothing here differentiates a graph, so each gradient is one link and the chain rule is the caller's. Signatures take tensors positionally and everything else as keywords with DirectMLX's defaults; multi-output operators return namedtuples; `FusedActivation.relu()` and its siblings replace bare `FusedActivation(OperatorType.ACTIVATION_RELU)`.
- Arithmetic on `Expression` comes from DirectMLX's overloads with three deviations: `%` is floored like Python's, `float / x` is corrected, and there are no in-place forms. Comparison operators stay identity-based, which is what makes an `Expression` a dict key.
- `CompiledOperator.temporary_size`, `persistent_size` and `descriptor_count` expose the operator's binding properties, which is the memory budget of a compiled graph.
- Errors carry the HRESULT by name, and a removed device reports why it was removed. Buffers grow by a fixed step past 256 MiB, because a single D3D12 buffer stops at 4 GiB.

## Prerequisites

- Windows with a DirectX 12 capable GPU
- Visual Studio with the **Desktop development with C++** workload, including Windows SDK **10.0.26100.0 or newer**. The bindings call DirectML feature level 6.2 and 6.3 entry points, which earlier headers do not expose.
- Python 3.8 or newer

CMake comes from PyPI as a build requirement; a Visual Studio Developer Command Prompt is not required.

## Build and installation

```powershell
git clone --recursive https://github.com/aisk/PyDirectML.git
cd PyDirectML

pip install .                    # the package
pip install .[samples]           # plus what samples/ needs
```

If the repository was cloned without `--recursive`, run `git submodule update --init --recursive` first; the build needs both the `pybind11` and `gpgmm` submodules.

To work on the bindings, install in editable mode: the extension is built into `directml/` and imported from the source tree.

```powershell
pip install -e .[samples,test]
python -m pytest tests
```

## Samples

`samples/matmul.py` is the smallest thing that works, and `samples/dtypes.py` runs one graph at float32 and at float16 and shows what `dispatch` refuses. `samples/mnist.py`, `squeezenet.py`, `mobilenet.py`, `superres.py` and `candy.py` come from upstream and run ONNX models from the `.npy` weights checked in beside them, under `samples/<model>_tensor_data/`. `candy.py` takes another style's directory as its second argument: `la_muse_tensor_data`, `mosaic_tensor_data` or `udnie_tensor_data`.

[`samples/sdxl/`](./samples/sdxl/) is Stable Diffusion XL built on these bindings: both CLIP text encoders, the UNet, the VAE, and the Euler sampler, running against the real SDXL weights. `python generate.py "a prompt"` produces a 1024×1024 image in 63 seconds on a Radeon RX 6800. It reads either the weights diffusers publishes or, with `--checkpoint`, a single-file checkpoint of the kind ComfyUI and A1111 use. The VAE and the text encoders are checked against a NumPy reference implementation that ships with it.
