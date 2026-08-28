//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

#pragma warning(push)
#pragma warning(disable:4238)

using namespace pydml;
using Microsoft::WRL::ComPtr;

// An adapter called the "Microsoft Basic Render Driver" is always present. This adapter is a render-only device that has no display outputs.
bool IsWarpAdapter(IDXGIAdapter1* pAdapter)
{
    DXGI_ADAPTER_DESC1 pDesc;
    ThrowIfFailed(pAdapter->GetDesc1(&pDesc));
    // See here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
    auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
    return isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
}

Device::Device(bool useGpu, bool useDebugLayer, DXGI_GPU_PREFERENCE gpuPreference) :
    m_useGpu(useGpu), m_gpuPreference(gpuPreference)
{
    // 
    // Create D3D12 resources
    //

    if (useDebugLayer)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
        }
    }

    ComPtr<IDXGIAdapter1> dxgiAdapter;
    if (m_useGpu)
    {
        ComPtr<IDXGIFactory6> spFactory;
        ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&spFactory)));
        UINT i = 0;
        while (spFactory->EnumAdapterByGpuPreference(i, m_gpuPreference, IID_PPV_ARGS(&dxgiAdapter)) != DXGI_ERROR_NOT_FOUND)
        {
            if (!IsWarpAdapter(dxgiAdapter.Get()))
            {
                break;
            }
            ++i;
        }
    }

    if (    !useGpu 
        ||  FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device))))
    {
        ComPtr<IDXGIFactory4> dxgiFactory;
        ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        ThrowIfFailed(dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&dxgiAdapter)));
        ThrowIfFailed(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_d3d12Device)));
    }

    // Lookup the hardware adapter used by the device.
    if (dxgiAdapter == nullptr){
        LUID adapterLUID = m_d3d12Device->GetAdapterLuid();
        ComPtr<IDXGIFactory1> dxgiFactory;
        ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)));
        ComPtr<IDXGIFactory4> dxgiFactory4;
        ThrowIfFailed(dxgiFactory.As(&dxgiFactory4));
        dxgiFactory4->EnumAdapterByLuid(adapterLUID, IID_PPV_ARGS(&dxgiAdapter));
    }

    D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
    ThrowIfFailed(m_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)));

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    ThrowIfFailed(m_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));

    gpgmm::d3d12::ALLOCATOR_DESC allocatorDesc = {};
    allocatorDesc.Adapter = dxgiAdapter;
    allocatorDesc.Device = m_d3d12Device;
    allocatorDesc.IsUMA = arch.UMA;
    allocatorDesc.ResourceHeapTier = options.ResourceHeapTier;
    ThrowIfFailed(gpgmm::d3d12::ResourceAllocator::CreateAllocator(allocatorDesc, &m_resourceAllocator));

    m_residencyManager = m_resourceAllocator->GetResidencyManager();

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ThrowIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_GRAPHICS_PPV_ARGS(m_commandQueue.GetAddressOf())));

    ThrowIfFailed(m_d3d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_GRAPHICS_PPV_ARGS(m_commandAllocator.GetAddressOf())));

    ThrowIfFailed(m_d3d12Device->CreateCommandList(
        0, // node mask
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr, // initial pipeline state
        IID_GRAPHICS_PPV_ARGS(m_commandList.GetAddressOf())));

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = 4; // One each for the input, output, persistent, and temporary resources
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapCpu.GetAddressOf())));

    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapGpu.GetAddressOf())));


    // 
    // Create DML resources
    //

    if (    !useDebugLayer 
        ||  FAILED(DMLCreateDevice(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_DEBUG, IID_PPV_ARGS(&m_dmlDevice))))
    {
        ThrowIfFailed(DMLCreateDevice(m_d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice)));
    }

    ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));
    ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_operatorInitializer)));
    ThrowIfFailed(m_dmlDevice->CreateBindingTable(nullptr, IID_PPV_ARGS(&m_bindingTable)));
}

Graph::Graph(std::shared_ptr<Device> device, dml::TensorPolicy tensorPolicy) :
    device(std::move(device)),
    graph(this->device->GetDevice(), std::move(tensorPolicy))
{
}

dml::Expression Graph::Input(uint32_t index, dml::TensorDesc desc)
{
    if (index != slots.size())
    {
        throw std::invalid_argument(
            "input index " + std::to_string(index) + " is out of order; the graph's next input is " +
            std::to_string(slots.size()));
    }

    auto expression = dml::InputTensor(graph, index, desc);
    slots.push_back(InputSlot {
        reinterpret_cast<uintptr_t>(expression.Impl()),
        expression.GetOutputDesc(),
        (desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML) != 0 });
    return expression;
}

CompiledOperator::CompiledOperator(
    Graph& graph,
    DML_EXECUTION_FLAGS flags,
    std::vector<dml::Expression> const& outputs
    ) :
    op(graph.graph.Compile(flags, outputs, static_cast<uint32_t>(graph.slots.size()))),
    device(graph.device),
    inputs(graph.slots)
{
    outputDescs.reserve(outputs.size());
    for (auto const& output : outputs)
    {
        outputDescs.push_back(output.GetOutputDesc());
    }
}

void Device::UploadStagedInputs(
    py::iterable staged,
    std::vector<pydml::InputSlot> const& slots,
    std::vector<DmlBufferBinding> const& bindings,
    gpgmm::d3d12::ResourceAllocation* inputsResource,
    uint64_t inputsResourceSize,
    bool owned
    )
{
    if (inputsResourceSize == 0)
    {
        return;
    }

    // Copy the data into the upload heap. The wrapper layer validated shapes and
    // dtypes and converts one array at a time as this loop pulls on `staged`, so
    // only one converted copy is alive at once; the guards below only keep a bad
    // caller of the private API from corrupting a neighboring tensor's bytes.
    byte* uploadHeapData = nullptr;
    ThrowIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

    try
    {
        for (py::handle item : staged)
        {
            auto pair = item.cast<std::pair<uint32_t, py::array>>();
            uint32_t index = pair.first;
            py::array const& array = pair.second;

            if (index >= slots.size() || slots[index].owned != owned)
            {
                throw std::invalid_argument(
                    "staged input " + std::to_string(index) + " is not bound in this phase");
            }

            auto const& binding = bindings[index];
            py::buffer_info info = array.request();
            auto sizeInBytes = static_cast<uint64_t>(info.size) * static_cast<uint64_t>(info.itemsize);

            if ((array.flags() & py::array::c_style) == 0 || sizeInBytes > binding.sizeInBytes)
            {
                throw std::invalid_argument(
                    "staged input " + std::to_string(index) +
                    " is not a contiguous array that fits its tensor");
            }

            byte* dest = uploadHeapData + binding.offset;
            memcpy(dest, info.ptr, static_cast<size_t>(sizeInBytes));

            // DirectML rounds a tensor's size up to a 4-byte boundary, so a
            // packed array can come up a few bytes short of it; the tail still
            // has to hold defined values.
            if (sizeInBytes < binding.sizeInBytes)
            {
                memset(dest + sizeInBytes, 0, static_cast<size_t>(binding.sizeInBytes - sizeInBytes));
            }
        }
    }
    catch (...)
    {
        m_uploadHeap->Unmap(0, nullptr);
        throw;
    }

    m_uploadHeap->Unmap(0, nullptr);

    // Record the copy from the upload heap into the inputs resource
    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            inputsResource->GetResource(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST)
        );

    m_commandList->CopyBufferRegion(inputsResource->GetResource(), 0, m_uploadHeap->GetResource(), 0, inputsResourceSize);

    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            inputsResource->GetResource(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        );
}

py::list Device::Dispatch(
    pydml::CompiledOperator& model,
    py::iterable staged
    )
{
    if (!model.initialized)
    {
        throw std::invalid_argument("the operator must be initialized before it is dispatched");
    }

    IDMLCompiledOperator* op = model.op.Get();

    std::vector<DmlBufferBinding> inputBindings(model.inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < model.inputs.size(); ++i)
    {
        auto& slot = model.inputs[i];

        // An input owned by DirectML lives in the persistent resource and is
        // not bound at execution.
        if (!slot.owned)
        {
            DmlBufferTensorDesc desc = *slot.desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            uint32_t requiredAlignment = std::max(desc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBindings[i].sizeInBytes = desc.totalTensorSizeInBytes;

            inputsResourceSize = inputBindings[i].offset + desc.totalTensorSizeInBytes;
        }
    }

    std::vector<DmlBufferBinding> outputBindings(model.outputDescs.size());
    uint64_t outputsResourceSize = 0;

    for (size_t i = 0; i < model.outputDescs.size(); ++i)
    {
        DmlBufferTensorDesc bufferDesc = *model.outputDescs[i].AsPtr<DML_BUFFER_TENSOR_DESC>();

        uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

        // Bind to the end of the outputs resource (with appropriate alignment)
        outputBindings[i].offset = RoundUpToMultiple(outputsResourceSize, (uint64_t)requiredAlignment);
        outputBindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

        outputsResourceSize = outputBindings[i].offset + outputBindings[i].sizeInBytes;
    }

    DML_BINDING_PROPERTIES bindingProps = op->GetBindingProperties();

    EnsureUploadHeapSize(inputsResourceSize);
    EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource);
    EnsureReadBackHeapSize(outputsResourceSize);
    EnsureCpuOrDefaultBufferSize(outputsResourceSize, m_outputsResource);
    EnsureDefaultBufferSize(bindingProps.TemporaryResourceSize, m_temporaryResource);
    EnsureDescriptorHeapSize(bindingProps.RequiredDescriptorCount);

    // Set up input and output bindings to point to their respective buffers
    for (auto& binding : inputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource->GetResource();
        }
    }

    for (auto& binding : outputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_outputsResource->GetResource();
        }
    }

    // The persistent resource should have already been initialized when the operator was initialized
    assert(model.persistentResource->GetResource()->GetDesc().Width >= bindingProps.PersistentResourceSize);

    // Upload inputs for execution
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource->GetResource(),
        m_temporaryResource->GetResource(),
        m_outputsResource->GetResource()
    };

    ClearGpuBuffers(buffersToClear);

    UploadStagedInputs(staged, model.inputs, inputBindings, m_inputsResource.Get(), inputsResourceSize, false);

    // Bind for execution
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->m_Heap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->m_Heap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    ThrowIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    // Bind inputs
    std::vector<DML_BINDING_DESC> inputBindingDescs(inputBindings.size());
    for (size_t i = 0; i < inputBindings.size(); ++i)
    {
        inputBindingDescs[i] = converter.ToBindingDesc(inputBindings[i]);
    }

    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    // Bind outputs
    std::vector<DML_BINDING_DESC> outputBindingDescs(outputBindings.size());
    for (size_t i = 0; i < outputBindings.size(); ++i)
    {
        outputBindingDescs[i] = converter.ToBindingDesc(outputBindings[i]);
    }

    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

    // Bind persistent/temporary resources
    if (bindingProps.PersistentResourceSize != 0)
    {
        DML_BUFFER_BINDING persistentBinding = { model.persistentResource->GetResource(), 0, bindingProps.PersistentResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &persistentBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }

    if (bindingProps.TemporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource->GetResource(), 0, bindingProps.TemporaryResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Record and execute commands, and wait for completion
    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap->m_Heap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), op, m_bindingTable.Get());
    RecordOutputReadBack(outputsResourceSize);
    ExecuteCommandListAndWait();

    // Read the output data back from the readback heap
    return DownloadFromReadBackHeap(outputsResourceSize, model.outputDescs, outputBindings);
}

void Device::RecordOutputReadBack(uint64_t outputsResourceSize)
{
    // Copy output to readback heap
    if (outputsResourceSize != 0)
    {
        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            );

        m_commandList->CopyBufferRegion(m_readbackHeap->GetResource(), 0, m_outputsResource->GetResource(), 0, outputsResourceSize);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource->GetResource(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }
}

py::list Device::DownloadFromReadBackHeap(
    uint64_t outputsResourceSize,
    std::vector<dml::TensorDesc> const& outputDescs,
    std::vector<DmlBufferBinding>& outputBindings
    )
{
    py::list outputs;

    if (outputsResourceSize != 0)
    {
        CD3DX12_RANGE readRange(0, static_cast<size_t>(outputsResourceSize));

        byte* readbackHeapData = nullptr;

        ThrowIfFailed(m_readbackHeap->Map(0, &readRange, reinterpret_cast<void**>(&readbackHeapData)));

        for (size_t i = 0; i < outputDescs.size(); ++i)
        {
            auto const& desc = outputDescs[i];
            auto const& type = GetDataType(desc.dataType);

            std::vector<py::ssize_t> shape(desc.sizes.begin(), desc.sizes.end());

            // numpy strides are in bytes. An empty vector makes py::array
            // compute packed C-order strides itself.
            std::vector<py::ssize_t> strides;
            if (desc.strides)
            {
                strides.reserve(desc.strides->size());
                for (auto stride : *desc.strides)
                {
                    strides.push_back(static_cast<py::ssize_t>(stride) * static_cast<py::ssize_t>(type.itemSize));
                }
            }

            // This constructor copies: it views the mapped heap through the
            // desc's shape and strides, then materializes a packed array that
            // owns its data.
            outputs.append(py::array(
                py::dtype(type.numpyName),
                shape,
                strides,
                readbackHeapData + outputBindings[i].offset));
        }

        m_readbackHeap->Unmap(0, nullptr);
    }

    return outputs;
}

void Device::Initialize(
    pydml::CompiledOperator& model,
    py::iterable staged
    )
{
    IDMLCompiledOperator* op = model.op.Get();

    // Allocate resources for initialization
    ThrowIfFailed(m_operatorInitializer->Reset(1, &op));

    DmlBufferArrayBinding inputBinding;
    inputBinding.bindings.resize(model.inputs.size());

    // Fill in the offsets and sizes for each binding, which will also tell us how big we need to make our buffer
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < model.inputs.size(); ++i)
    {
        auto& slot = model.inputs[i];

        // Only the inputs owned by DirectML are bound at initialize.
        if (slot.owned)
        {
            DmlBufferTensorDesc bufferDesc = *slot.desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBinding.bindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBinding.bindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

            inputsResourceSize = inputBinding.bindings[i].offset + bufferDesc.totalTensorSizeInBytes;
        }
    }

    uint64_t temporaryResourceSize = m_operatorInitializer->GetBindingProperties().TemporaryResourceSize;
    uint64_t persistentResourceSize = op->GetBindingProperties().PersistentResourceSize;
    uint32_t descriptorHeapSize = m_operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    EnsureUploadHeapSize(inputsResourceSize);
    EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource);
    EnsureDefaultBufferSize(temporaryResourceSize, m_temporaryResource);
    model.allocator = m_resourceAllocator;
    EnsureDefaultBufferSize(persistentResourceSize, model.persistentResource);
    EnsureDescriptorHeapSize(descriptorHeapSize);

    model.persistentResourceSize = persistentResourceSize;

    // Set up the bindings to point to our input resource
    for (auto& binding : inputBinding.bindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource->GetResource();
        }
    }

    // Upload inputs for initialization
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource->GetResource(),
        m_temporaryResource->GetResource(),
        model.persistentResource->GetResource()
    };

    ClearGpuBuffers(buffersToClear);

    UploadStagedInputs(staged, model.inputs, inputBinding.bindings, m_inputsResource.Get(), inputsResourceSize, true);

    // Bind for initialization
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_operatorInitializer.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->m_Heap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->m_Heap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = descriptorHeapSize;

    ThrowIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    DML_BINDING_DESC inputBindingDesc = converter.ToBindingDesc(inputBinding);
    m_bindingTable->BindInputs(1, &inputBindingDesc);

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING outputBinding = { model.persistentResource->GetResource(), 0, persistentResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &outputBinding };
        m_bindingTable->BindOutputs(1, &desc);
    }

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource->GetResource(), 0, temporaryResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&desc);
    }

    // Record and execute commands, and wait for completion
    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap->m_Heap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), m_operatorInitializer.Get(), m_bindingTable.Get());
    ExecuteCommandListAndWait();

    model.initialized = true;
}

void Device::ExecuteCommandListAndWait()
{
    ThrowIfFailed(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    if (m_residencyManager != nullptr){
        gpgmm::d3d12::ResidencySet* residencySets[] = { &m_residencySet };
        m_residencyManager->ExecuteCommandLists(m_commandQueue.Get(), commandLists, residencySets, ARRAYSIZE(commandLists));
    } else {
        m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    }

    WaitForQueueToComplete(m_commandQueue.Get());

    ThrowIfFailed(m_commandAllocator->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

    if (m_residencyManager != nullptr){
        ThrowIfFailed(m_residencySet.Reset());
    }
}

void Device::EnsureUploadHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_uploadHeap ? m_uploadHeap->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = GrowBufferSize(requestedSizeInBytes);

    if (newSize != existingSize)
    {
        m_uploadHeap = nullptr;
        m_uploadHeap = CreateResource(
            m_resourceAllocator.Get(),
            CD3DX12_RESOURCE_DESC::Buffer(newSize),
            CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_RESOURCE_STATE_GENERIC_READ
            );
    }
}

void Device::EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    if (m_useCpuCustomHeapResources)
    {
        EnsureCpuBufferSize(requestedSizeInBytes, buffer);
    }
    else
    {
        EnsureDefaultBufferSize(requestedSizeInBytes, buffer);
    }
}

void Device::EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = GrowBufferSize(requestedSizeInBytes);

    if (newSize != existingSize)
    {
        buffer = nullptr;
        buffer = CreateCpuCustomBuffer(m_resourceAllocator.Get(), newSize);
    }

    buffer->UpdateResidency(&m_residencySet);
}

void Device::EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer)
{
    uint64_t existingSize = buffer ? buffer->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = GrowBufferSize(requestedSizeInBytes);

    if (newSize != existingSize)
    {
        buffer = nullptr;
        buffer = CreateDefaultBuffer(m_resourceAllocator.Get(), newSize);
    }

    buffer->UpdateResidency(&m_residencySet);
}

void Device::EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors)
{
    uint32_t existingSize = m_descriptorHeap ? m_descriptorHeap->m_Heap->GetDesc().NumDescriptors : 0;
    uint32_t newSize = RoundUpToPow2(requestedSizeInDescriptors); // ensures geometric growth

    if (newSize != existingSize)
    {
        if (m_residencyManager != nullptr){
            m_residencyManager->UnlockHeap(m_descriptorHeap.get());
        }

        m_descriptorHeap = nullptr;

        if (m_residencyManager != nullptr){
            ThrowIfFailed(m_residencyManager->Evict(newSize, DXGI_MEMORY_SEGMENT_GROUP_LOCAL));
        }

        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = newSize;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        ComPtr<ID3D12DescriptorHeap> d3d12DescriptorHeap;
        ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(d3d12DescriptorHeap.GetAddressOf())));
    
        m_descriptorHeap = std::make_unique<SVDescriptorHeap>(std::move(d3d12DescriptorHeap), newSize);

        if (m_residencyManager != nullptr){
            ThrowIfFailed(m_residencyManager->InsertHeap(m_descriptorHeap.get()));
            ThrowIfFailed(m_residencyManager->LockHeap(m_descriptorHeap.get()));
        }
    }
}

void Device::EnsureReadBackHeapSize(uint64_t requestedSizeInBytes)
{
    uint64_t existingSize = m_readbackHeap ? m_readbackHeap->GetResource()->GetDesc().Width : 0;
    uint64_t newSize = GrowBufferSize(requestedSizeInBytes);

    if (newSize != existingSize)
    {
        m_readbackHeap = nullptr;
        m_readbackHeap = CreateReadBackBuffer(m_resourceAllocator.Get(), newSize);
    }
    
    m_readbackHeap->UpdateResidency(&m_residencySet);
}

void Device::ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers)
{
    static const uint32_t ClearValue = static_cast<uint32_t>(-1);

    // The number of buffers we can clear at once is limited by the size of our descriptor heap
    assert(static_cast<uint32_t>(buffers.size()) <= m_clearUavDescriptorHeapCpu->GetDesc().NumDescriptors);

    uint32_t descriptorOffset = 0;
    for (ID3D12Resource* buffer : buffers)
    {
        if (!buffer)
        {
            // Nothing to clear; these buffers are lazily-initialized
            continue;
        }

        FillGpuBuffer(
            m_commandList.Get(),
            m_clearUavDescriptorHeapCpu.Get(),
            m_clearUavDescriptorHeapGpu.Get(),
            descriptorOffset,
            buffer,
            ClearValue);

        ++descriptorOffset;
    }

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
}

#pragma warning(pop)