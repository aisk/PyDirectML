//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#define NOMINMAX
#include <cassert>
#include <stdexcept>
#include <optional>
#include <string>
#include <functional>
#include <numeric>

#ifdef __cpp_lib_span
#include <span>
#endif

#include <Windows.h>
#include <d3d12.h>

// ToDo: dxgi isn't available in WSL.
#include <dxgi1_5.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>

#include <initguid.h>
#include <wrl/client.h>
#include <wrl/implements.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#define DML_TARGET_VERSION_USE_LATEST 1
#include <DirectML.h>
// DirectMLX guards several locals with assert() only, so they read as unused
// once NDEBUG compiles the asserts out.
#pragma warning(push)
#pragma warning(disable: 4189) // local variable is initialized but not referenced
#include <DirectMLX.h>
#pragma warning(pop)

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
#include "d3dx12.h"
#include "util.h"
#include "model.h"
#include "typeconvert.h"
#include "device.h"
