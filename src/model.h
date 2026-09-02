//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

namespace pydml
{
    class Device;

    // One input of a graph: which expression it is, what tensor it wants, and
    // whether DirectML owns its data. `key` is the expression's NodeOutput
    // pointer, kept only as an identity to match dict keys against -- it is
    // never dereferenced, so it stays valid as a key after the graph is gone.
    struct InputSlot
    {
        uintptr_t key;
        dml::TensorDesc desc;
        bool owned;
    };

    // A dml::Graph plus the record of its inputs, which is what lets a dict
    // keyed by Expression replace bindings matched by position: every input's
    // index is assigned here and snapshotted into the CompiledOperator.
    struct Graph
    {
        Graph(std::shared_ptr<Device> device, dml::TensorPolicy tensorPolicy);

        // Add an input at the next free index.
        dml::Expression Input(dml::TensorDesc desc);

        std::shared_ptr<Device> device;
        dml::Graph graph;
        std::vector<InputSlot> slots;
    };

    struct CompiledOperator
    {
        CompiledOperator(
            Graph& graph,
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression> const& outputs
            );

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;

        // The device that compiled this operator, and snapshots of what the
        // graph knew. Together they make the operator self-contained, so the
        // graph can be dropped as soon as compile returns.
        std::shared_ptr<Device> device;
        std::vector<InputSlot> inputs;
        std::vector<dml::TensorDesc> outputDescs;

        // The allocator that made persistentResource. gpgmm hands the memory back
        // to it on release, and Python is free to destroy the Device before the
        // operator, so the allocation keeps its allocator alive. Declared before
        // the allocation so it is destroyed last.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocator> allocator;

        // Written by Device::Initialize. DirectML folds the DML_TENSOR_FLAG_OWNED_BY_DML
        // inputs into this buffer, in whatever layout the operator wants, and
        // reads them from here on every dispatch.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> persistentResource;
        bool initialized = false;
    };

    // A tensor that lives on the GPU: a DEFAULT-heap resource plus the desc
    // that says how to read it. Dispatch hands one out per output when asked
    // not to read back, and accepts one wherever it accepts an array, bound
    // directly with no upload. The device is declared first so it outlives
    // the allocation it made.
    struct Buffer
    {
        Buffer(
            std::shared_ptr<Device> device,
            dml::TensorDesc desc,
            Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> resource
            ) :
            device(std::move(device)),
            desc(std::move(desc)),
            resource(std::move(resource))
        {
        }

        std::shared_ptr<Device> device;
        dml::TensorDesc desc;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> resource;

        uint64_t SizeInBytes() const
        {
            return desc.totalTensorSizeInBytes;
        }
    };
}
