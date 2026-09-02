cmake_minimum_required(VERSION 3.15)

project(directml LANGUAGES CXX)

# DirectML.h and DirectML.lib ship with the Windows SDK, so they resolve through
# the default include and library paths. DirectMLX.h and d3dx12.h are header-only
# and vendored under third_party/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(gpgmm)
add_subdirectory(pybind11)

pybind11_add_module(
    _core
    src/util.cpp
    src/module.cpp
    src/device.cpp
    )

target_compile_options(_core PRIVATE /W4 /WX)
target_include_directories(_core PRIVATE third_party)
target_link_libraries(_core PRIVATE gpgmm gpgmm_public_config dxgi.lib d3d12.lib directml.lib)
