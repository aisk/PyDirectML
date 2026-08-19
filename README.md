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
- `activation_soft_max` is spelled `activation_softmax` and takes an `axes` argument, binding `DML_ACTIVATION_SOFTMAX1`. The old binding normalized a flattened 2-D view and could not express softmax over the last axis of a 4-D tensor, which attention needs.

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
```

If the repository was cloned without `--recursive`, run `git submodule update --init --recursive` first. The build needs both the `pybind11` and `gpgmm` submodules.

## Usage
In a Python file, import the module

    import directml

The extension links against the Windows SDK import library and loads the `DirectML.dll` that ships with Windows; nothing is bundled or redistributed. To run against a different build, place that `DirectML.dll` next to the installed `.pyd` in `site-packages`.
