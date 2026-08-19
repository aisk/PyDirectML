# Python Binding for DirectML Samples

> Forked from the `Python/` directory of [microsoft/DirectML](https://github.com/microsoft/DirectML), which is no longer maintained. `third_party/DirectMLX.h` comes from that repository's `Libraries/` directory. Licensed under MIT, see [LICENSE](./LICENSE).

PyDirectML is an open source Python binding library for DirectML written to facilitate DirectML sample authoring in Python. It provides the following capabilities to Python sample authors:
- Simplified DirectML graph authoring and compilation with operator composition
- Wrapper of DirectML device and resource management
- Binding support through NumPy arrays

## Differences from upstream

- The extension module is imported as **`directml`**, not `pydirectml`.
- `DirectML.h` and `DirectML.lib` come from the Windows SDK. Upstream downloaded the `microsoft.ai.directml` NuGet package at build time and shipped its `DirectML.dll` inside the wheel, where nothing ever loaded it.
- `average_pooling` takes `dilations` and `mean_variance_normalization` takes `normalize_mean`, in the position the DirectMLX signature gives them. The samples are updated to match.
- `activation_soft_max` is spelled `activation_softmax` and takes an `axes` argument, binding `DML_ACTIVATION_SOFTMAX1`. The old binding normalized a flattened 2-D view and could not express softmax over the last axis of a 4-D tensor, which attention needs. `activation_gelu` is bound as well.
- A tensor's declared `TensorDataType` is honored on both ends. `Binding` converts what you hand it to that type and refuses conversions that cross a dtype kind unsafely; results come back as that type instead of always as float32. Upstream forced float32 in and read float32 out, which reads out of bounds for any narrower type. Half precision works: see `samples/dtypes.py`.
- `Device.initialize` and `Device.dispatch` are separate, and the persistent resource belongs to the model rather than to the device. `compute` initializes on first use and dispatches after that, so a tensor flagged `OWNED_BY_DML` is uploaded once and then stays on the GPU. Upstream re-ran initialization on every `compute`, re-uploading every weight each time.
- Buffers grow by a fixed step once they pass 256 MiB instead of doubling forever. Doubling turned a 2.1 GiB request into a 4 GiB single resource, which removes the device (`DXGI_ERROR_DEVICE_REMOVED`) rather than failing the allocation. `ThrowIfFailed` also reports the HRESULT now; it used to throw a bare `std::exception`, which reached Python as "Unknown exception".

## Prerequisites

- Windows with a DirectX 12 capable GPU
- Visual Studio with the **Desktop development with C++** workload, including Windows SDK **10.0.26100.0 or newer**. The bindings call DirectML feature level 6.2 and 6.3 entry points, which earlier headers do not expose.
- CMake 3.15 or newer, on `PATH`. Visual Studio bundles one under `<VS install>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.
- Python 3.6 or newer, with `setuptools` and `wheel`

A Visual Studio Developer Command Prompt is *not* required, and the build needs no network access.

## Build and Installation

```powershell
git clone --recursive https://github.com/aisk/PyDirectML.git
cd PyDirectML

# pybind11 v2.10 declares cmake_minimum_required(VERSION 3.4), which CMake 4.x refuses. Not needed with CMake 3.x.
$env:CMAKE_POLICY_VERSION_MINIMUM = '3.5'

pip install . --no-build-isolation

# To run the samples, install the extra dependencies as well.
pip install .[dev] --no-build-isolation
```

The samples under `samples/` need NumPy, the image samples additionally need Pillow, and `samples/sdxl/` needs `safetensors`, `huggingface_hub` and `transformers` (for its CLIP tokenizer only — no PyTorch). The `dev` extra pulls in all of them.

## Samples

`samples/matmul.py` is the smallest thing that works, and `samples/dtypes.py` runs one graph at float32 and at float16 and shows what `Binding` refuses. `samples/mnist.py`, `squeezenet.py`, `mobilenet.py`, `candy.py` and `superres.py` come from upstream and run ONNX models from the `.npy` weights checked in beside them.

[`samples/sdxl/`](./samples/sdxl/) is a Stable Diffusion XL pipeline built on these bindings: the VAE in both directions, both CLIP text encoders, and the Euler sampler, all against the real SDXL weights and all checked against a NumPy reference implementation that ships with it. Only the UNet is still missing.

If the repository was cloned without `--recursive`, run `git submodule update --init --recursive` first. The build needs both the `pybind11` and `gpgmm` submodules.

## Usage
In a Python file, import the module

    import directml

The extension links against the Windows SDK import library and loads the `DirectML.dll` that ships with Windows; nothing is bundled or redistributed. To run against a different build, place that `DirectML.dll` next to the installed `.pyd` in `site-packages`.
