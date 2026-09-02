//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <gpgmm_d3d12.h>

// The HRESULTs worth spelling out. Device removal is the one this hits in
// practice, and its reason code says whether the GPU hung, was reset out from
// under us, or the driver failed something on its own. E_INVALIDARG is what a
// graph DirectML refuses to compile comes back as -- a shape mismatch, say.
inline const char* HresultName(HRESULT hr)
{
    switch (hr)
    {
    case DXGI_ERROR_DEVICE_HUNG:           return "DXGI_ERROR_DEVICE_HUNG";
    case DXGI_ERROR_DEVICE_REMOVED:        return "DXGI_ERROR_DEVICE_REMOVED";
    case DXGI_ERROR_DEVICE_RESET:          return "DXGI_ERROR_DEVICE_RESET";
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR: return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
    case DXGI_ERROR_INVALID_CALL:          return "DXGI_ERROR_INVALID_CALL";
    case E_INVALIDARG:                     return "E_INVALIDARG";
    case E_OUTOFMEMORY:                    return "E_OUTOFMEMORY";
    default:                               return nullptr;
    }
}

inline std::string DescribeHresult(HRESULT hr)
{
    char code[24];
    snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned int>(hr));

    auto name = HresultName(hr);
    return name ? std::string(name) + " (" + code + ")" : "HRESULT " + std::string(code);
}

// A removed device is not reported by whatever caused it. The next call to touch
// the device fails with DXGI_ERROR_DEVICE_REMOVED, which says nothing about why;
// only the device itself knows, and only until it is released.
inline void ThrowIfDeviceRemoved(ID3D12Device* device)
{
    auto reason = device->GetDeviceRemovedReason();
    if (FAILED(reason))
    {
        throw std::runtime_error("the device was removed: " + DescribeHresult(reason));
    }
}

inline void ThrowIfFailed(HRESULT hr)
{
    if (FAILED(hr))
    {
        throw std::runtime_error(DescribeHresult(hr));
    }
}

// Same, for calls made against a device that can be asked why it went away.
inline void ThrowIfFailed(HRESULT hr, ID3D12Device* device)
{
    if (hr == DXGI_ERROR_DEVICE_REMOVED && device)
    {
        ThrowIfDeviceRemoved(device);
    }
    ThrowIfFailed(hr);
}

// How a DML_TENSOR_DATA_TYPE is read back into numpy: the element size, and
// the dtype name py::dtype takes. float16 has no C++ type, only a name.
struct DmlDataType
{
    size_t itemSize;
    const char* numpyName;
};

inline DmlDataType const& GetDataType(DML_TENSOR_DATA_TYPE dataType)
{
    static const DmlDataType types[] =
    {
        { 0, ""        },  // DML_TENSOR_DATA_TYPE_UNKNOWN
        { 4, "float32" },
        { 2, "float16" },
        { 4, "uint32"  },
        { 2, "uint16"  },
        { 1, "uint8"   },
        { 4, "int32"   },
        { 2, "int16"   },
        { 1, "int8"    },
        { 8, "float64" },
        { 8, "uint64"  },
        { 8, "int64"   },
    };

    auto index = static_cast<size_t>(dataType);
    if (index == 0 || index >= std::size(types))
    {
        throw std::invalid_argument("unsupported tensor data type " + std::to_string(index));
    }
    return types[index];
}

inline Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> CreateResource(
    gpgmm::d3d12::ResourceAllocator* resourceAllocator,
    const D3D12_RESOURCE_DESC& resourceDesc,
    const D3D12_HEAP_PROPERTIES& heapProperties,
    D3D12_RESOURCE_STATES initialState
    )
{
    gpgmm::d3d12::ALLOCATION_DESC allocationDesc = {};
    allocationDesc.HeapType = heapProperties.Type;

    Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> resource;
    ThrowIfFailed(resourceAllocator->CreateResource(
        allocationDesc,
        resourceDesc,
        initialState,
        nullptr,
        resource.GetAddressOf()));

    return resource;
}

inline Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> CreateDefaultBuffer(
    gpgmm::d3d12::ResourceAllocator* resourceAllocator,
    UINT64 sizeInBytes,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    return CreateResource(
        resourceAllocator,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, flags),
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
}

inline Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> CreateReadBackBuffer(gpgmm::d3d12::ResourceAllocator* resourceAllocator, UINT64 sizeInBytes)
{
    return CreateResource(
        resourceAllocator,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes),
        CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_RESOURCE_STATE_COPY_DEST
        );
}

void WaitForQueueToComplete(ID3D12CommandQueue* queue);

inline std::string UintVectorToString(std::vector<uint32_t> const& v)
{
    if (v.empty())
        return std::string();

    return std::accumulate(v.begin() + 1, v.end(), std::to_string(v[0]),
        [](std::string const& a, int b) {
            return a + ',' + std::to_string(b);
        });
}

template <typename T>
T RoundUpToMultiple(T value, T multiple)
{
    static_assert(std::is_integral_v<T>);

    T remainder = value % multiple;
    if (remainder != 0)
    {
        value += multiple - remainder;
    }

    return value;
}

template <typename T>
T RoundUpToPow2(T value)
{
    static_assert(std::is_integral_v<T>);

    if (value >= std::numeric_limits<T>::max() / 2)
    {
        ThrowIfFailed(E_INVALIDARG); // overflow
    }

    T pow2 = 1;
    while (pow2 < value)
    {
        pow2 *= 2;
    }

    return pow2;
}

// How large a buffer to allocate for a request: doubling up to 256 MiB, then a
// fixed step. Doubling past that wastes hundreds of megabytes and rounds a
// 2.1 GiB request up to a 4 GiB single resource, which removes the device
// (see docs/api-design.md on the 4 GiB limit).
inline uint64_t GrowBufferSize(uint64_t requestedSizeInBytes)
{
    constexpr uint64_t minimumSizeInBytes = 65536;         // 64 KiB
    constexpr uint64_t stepSizeInBytes = 256ull << 20;     // 256 MiB

    if (requestedSizeInBytes <= stepSizeInBytes)
    {
        return std::max(RoundUpToPow2(requestedSizeInBytes), minimumSizeInBytes);
    }
    return RoundUpToMultiple(requestedSizeInBytes, stepSizeInBytes);
}
