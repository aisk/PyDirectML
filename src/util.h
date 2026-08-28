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
        // Without the code this surfaces in Python as "Unknown exception", which
        // says nothing about whether a device call failed or an allocation did.
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

inline void ThrowIfNull(void* p)
{
    if (!p)
        throw std::exception();
}

// How a DML_TENSOR_DATA_TYPE is spelled elsewhere. `format` is a buffer protocol
// format string (PEP 3118), which is what NumPy reads a TensorData through;
// `numpyName` is the dtype constructor argument used to check what a Binding was
// handed. 'e' is half float, which has a buffer format but no C++ type.
struct DmlDataType
{
    size_t itemSize;
    const char* format;
    const char* numpyName;
};

inline DmlDataType const& GetDataType(DML_TENSOR_DATA_TYPE dataType)
{
    static const DmlDataType types[] =
    {
        { 0, "",  ""        },  // DML_TENSOR_DATA_TYPE_UNKNOWN
        { 4, "f", "float32" },
        { 2, "e", "float16" },
        { 4, "I", "uint32"  },
        { 2, "H", "uint16"  },
        { 1, "B", "uint8"   },
        { 4, "i", "int32"   },
        { 2, "h", "int16"   },
        { 1, "b", "int8"    },
        { 8, "d", "float64" },
        { 8, "Q", "uint64"  },
        { 8, "q", "int64"   },
    };

    auto index = static_cast<size_t>(dataType);
    if (index == 0 || index >= std::size(types))
    {
        throw std::invalid_argument("unsupported tensor data type " + std::to_string(index));
    }
    return types[index];
}

// DML_BUFFER_TENSOR_DESC (DML_TENSOR_TYPE_BUFFER)
struct DmlBufferTensorDesc
{
    DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE;
    std::vector<uint32_t> sizes;
    std::optional<std::vector<uint32_t>> strides;
    uint64_t totalTensorSizeInBytes = 0;
    uint32_t guaranteedBaseOffsetAlignment = 0;

    DmlBufferTensorDesc() = default;

    /*implicit*/ DmlBufferTensorDesc(const DML_BUFFER_TENSOR_DESC& desc)
        : dataType(desc.DataType),
        flags(desc.Flags),
        sizes(desc.Sizes, desc.Sizes + desc.DimensionCount),
        totalTensorSizeInBytes(desc.TotalTensorSizeInBytes),
        guaranteedBaseOffsetAlignment(desc.GuaranteedBaseOffsetAlignment)
    {
        if (desc.Strides)
        {
            strides.emplace(desc.Strides, desc.Strides + desc.DimensionCount);
        }
    }

    // Constructs a DmlBufferTensorDesc from a generic DML_TENSOR_DESC. The type must be DML_TENSOR_TYPE_BUFFER.
    /*implicit*/ DmlBufferTensorDesc(const DML_TENSOR_DESC& desc)
        : DmlBufferTensorDesc(*static_cast<const DML_BUFFER_TENSOR_DESC*>(desc.Desc))
    {
        assert(desc.Type == DML_TENSOR_TYPE_BUFFER);
    }

    uint32_t GetDimensionCount() const
    {
        assert(!strides || strides->size() == sizes.size());
        return static_cast<uint32_t>(sizes.size());
    }

    operator DML_BUFFER_TENSOR_DESC() const
    {
        DML_BUFFER_TENSOR_DESC bufferTensorDesc;
        bufferTensorDesc.DataType = dataType;
        bufferTensorDesc.DimensionCount = GetDimensionCount();
        bufferTensorDesc.Flags = flags;
        bufferTensorDesc.GuaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
        bufferTensorDesc.Sizes = sizes.data();
        bufferTensorDesc.Strides = strides ? strides->data() : nullptr;
        bufferTensorDesc.TotalTensorSizeInBytes = totalTensorSizeInBytes;

        return bufferTensorDesc;
    }
};

// (DML_BINDING_TYPE_NONE)
struct DmlNoneBinding
{
};

// DML_BUFFER_BINDING (DML_BINDING_TYPE_BUFFER)
struct DmlBufferBinding
{
    ID3D12Resource* buffer;
    uint64_t offset;
    uint64_t sizeInBytes;

    DmlBufferBinding() = default;

    /*implicit*/ DmlBufferBinding(const DML_BUFFER_BINDING& desc)
        : buffer(desc.Buffer),
        offset(desc.Offset),
        sizeInBytes(desc.SizeInBytes)
    {
    }
};

// DML_BUFFER_ARRAY_BINDING (DML_BINDING_TYPE_BUFFER_ARRAY)
struct DmlBufferArrayBinding
{
    std::vector<DmlBufferBinding> bindings;

    DmlBufferArrayBinding() = default;

    /*implicit*/ DmlBufferArrayBinding(const DML_BUFFER_ARRAY_BINDING& desc)
        : bindings(desc.Bindings, desc.Bindings + desc.BindingCount)
    {
    }
};

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

inline Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> CreateCpuCustomBuffer(
    gpgmm::d3d12::ResourceAllocator* resourceAllocator,
    UINT64 sizeInBytes,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    D3D12_HEAP_PROPERTIES heapProperties = {
        D3D12_HEAP_TYPE_CUSTOM,
        D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
        D3D12_MEMORY_POOL_L0,
        0,
        0
    };

    return CreateResource(
        resourceAllocator,
        CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, flags),
        heapProperties,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
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

void FillGpuBuffer(
    ID3D12GraphicsCommandList* commandList,
    ID3D12DescriptorHeap* descriptorHeapCpuVisible,
    ID3D12DescriptorHeap* descriptorHeapGpuVisible,
    uint32_t descriptorOffset,
    ID3D12Resource* buffer,
    uint32_t value
    );

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

// Rounds up a value to the nearest power of two
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

// How large a buffer to actually allocate for a request. Doubling keeps repeated
// small growth from reallocating every time, but past a gigabyte it both wastes
// hundreds of megabytes and walks into a wall: a 2.1 GiB request rounds to a
// 4 GiB single resource, which removes the device outright (DXGI_ERROR_DEVICE_REMOVED)
// on at least some drivers. Above the step size, grow by a fixed amount instead.
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
