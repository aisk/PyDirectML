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
// DirectMLX's own failure macro throws the failing expression's text with no
// HRESULT in it, so a graph DirectML rejects surfaced as a bare
// "m_device->CreateOperator(...)". Route it through DescribeHresult (defined
// in util.h, declared here because the macro expands before that include) so
// the error carries E_INVALIDARG or whatever the code was, like every other
// throw in the bindings.
std::string DescribeHresult(HRESULT hr);
#define DMLX_THROW_IF_FAILED(_hr) \
    do { HRESULT _dmlx_hr = (_hr); if (FAILED(_dmlx_hr)) { \
        throw std::runtime_error(std::string(#_hr) + " failed: " + DescribeHresult(_dmlx_hr)); \
    } } while (0)
#define DMLX_THROW(_hr) throw std::runtime_error(DescribeHresult(_hr))
// DirectMLX guards several locals with assert() only, so they read as unused
// once NDEBUG compiles the asserts out.
#pragma warning(push)
#pragma warning(disable: 4189) // local variable is initialized but not referenced
#include <DirectMLX.h>
#pragma warning(pop)

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
#include "d3dx12.h"
#include "util.h"
#include "attention.h"
#include "model.h"
#include "typeconvert.h"
#include "device.h"
