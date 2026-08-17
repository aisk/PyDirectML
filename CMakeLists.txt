cmake_minimum_required(VERSION 3.15)

project(directml)

# DirectML.h and DirectML.lib ship with the Windows SDK, so they resolve through
# the default include and library paths. DirectMLX.h is header-only and vendored
# under third_party/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(MSVC)
if(MSVC_VERSION GREATER_EQUAL 1920)
# Use C++17 on VS 16.0 (aka 2019) and later but with relaxed conformance check
set(PYBIND11_CPP_STANDARD /std:c++17 /permissive)
else()
# Use the latest C++ standard available for newer features
set(PYBIND11_CPP_STANDARD /std:c++latest)
endif()

# ssize_t does not exist for MSVC but SSIZE_T is available in BaseTsd.
add_definitions(-Dssize_t=SSIZE_T)
set(CMAKE_REQUIRED_DEFINITIONS -Dssize_t=SSIZE_T)
set(CMAKE_EXTRA_INCLUDE_FILES BaseTsd.h)

endif()

add_subdirectory(gpgmm)
add_subdirectory(pybind11)

pybind11_add_module(
    directml
    src/util.cpp
    src/module.cpp
    src/device.cpp
    )

target_compile_options(directml PRIVATE /W4 /WX)
target_include_directories(directml PRIVATE pybind11/include gpgmm/src/include gpgmm/ third_party)
target_link_libraries(directml PRIVATE gpgmm dxgi.lib d3d12.lib directml.lib)
