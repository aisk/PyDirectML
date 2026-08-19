//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    struct CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;

        // The allocator that made persistentResource. gpgmm hands the memory back
        // to it on release, and Python is free to destroy the Device before the
        // Model, so the allocation keeps its allocator alive. Declared first so it
        // is destroyed last.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocator> allocator;

        // Written by Device::Initialize. DirectML folds the DML_TENSOR_FLAG_OWNED_BY_DML
        // inputs into this buffer -- reordered into whatever layout the operator
        // wants -- and reads them from here on every dispatch. Owning it per
        // operator rather than per device is what lets those weights stay on the
        // GPU across dispatches instead of being re-uploaded each time.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> persistentResource;
        uint64_t persistentResourceSize = 0;
        bool initialized = false;
    };

    struct TensorData
    {
        TensorData(py::buffer_info const& info) :
            itemSize(info.itemsize),
            format(info.format),
            dimensions(info.ndim),
            shape(info.shape),
            strides(info.strides)
        {
            auto sizeInBytes = Size();
            buffer.resize(sizeInBytes);
            memcpy(buffer.data(), info.ptr, sizeInBytes);

            // Numpy strides use bytes.
            std::for_each(strides.begin(), strides.end(), [=](auto& i) {i *= itemSize; });
        }

        TensorData(dml::TensorDesc* desc) :
            itemSize(GetDataType(desc->dataType).itemSize),
            format(GetDataType(desc->dataType).format),
            dimensions(desc->sizes.size())
        {
            for (auto size : desc->sizes)
            {
                shape.push_back(static_cast<ssize_t>(size));
            }

            if (desc->strides)
            {
                for (auto stride : *desc->strides)
                {
                    strides.push_back(static_cast<ssize_t>(stride));
                }
            }
            else
            {
                // Use default descending packed strides.
                strides.resize(shape.size());
                ssize_t stride = 1;
                for (size_t i = strides.size(); i-- > 0; )
                {
                    strides[i] = stride;
                    stride *= shape[i];
                }
            }
            // Numpy strides use bytes.
            std::for_each(strides.begin(), strides.end(), [=](auto& i) {i *= itemSize; });

            buffer.resize(static_cast<size_t>(desc->totalTensorSizeInBytes));
        }

        TensorData() {}

        // Drop the CPU copy. For an OWNED_BY_DML input this is safe once the
        // model has been initialized: DirectML has folded the data into the
        // model's persistent resource and never reads this buffer again. At the
        // scale of a diffusion UNet this copy is gigabytes.
        void Release()
        {
            buffer.clear();
            buffer.shrink_to_fit();
        }

        void* Get() const { return static_cast<void*>(const_cast<byte*>(buffer.data())); }

        size_t Size() const
        {
            size_t size = 1;

            for (auto length : shape)
            {
                size *= length;
            }

            return size * itemSize;
        }

        std::vector<byte> buffer;
        size_t itemSize;
        std::string format;
        size_t dimensions;
        std::vector<ssize_t> shape;
        std::vector<ssize_t> strides;
    };

    struct Binding
    {
        explicit Binding(dml::Expression& expression, py::buffer_info const& info)
            :   desc(expression.GetOutputDesc()),
                data(info)
        {
            auto required = static_cast<size_t>(desc.AsPtr<DML_BUFFER_TENSOR_DESC>()->TotalTensorSizeInBytes);

            if (data.buffer.size() > required)
            {
                throw std::invalid_argument(
                    "array of " + std::to_string(data.buffer.size()) +
                    " bytes does not fit a tensor of " + std::to_string(required) + " bytes");
            }

            // DirectML rounds a tensor's size up to a 4-byte boundary, so a packed
            // array can come up a few bytes short. Device::Dispatch copies
            // TotalTensorSizeInBytes out of this buffer, so pad it rather than let
            // that copy read past the end.
            data.buffer.resize(required);
        }

        Binding() = default;

        dml::TensorDesc desc;
        TensorData data;
    };
}
