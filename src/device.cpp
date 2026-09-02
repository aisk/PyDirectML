//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

// CD3DX12_RESOURCE_BARRIER::Transition() returns a temporary whose address is
// taken; MSVC flags that as a non-standard extension.
#pragma warning(push)
#pragma warning(disable:4238)

using namespace pydml;
using Microsoft::WRL::ComPtr;

// The "Microsoft Basic Render Driver" is always present and has no display
// outputs; skip it when looking for a real GPU.
bool IsWarpAdapter(IDXGIAdapter1* pAdapter)
{
    DXGI_ADAPTER_DESC1 pDesc;
    ThrowIfFailed(pAdapter->GetDesc1(&pDesc));
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
    auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
    return isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
}

Device::Device(bool useGpu, bool useDebugLayer) :
    m_useGpu(useGpu)
{
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
        while (spFactory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_UNSPECIFIED, IID_PPV_ARGS(&dxgiAdapter)) != DXGI_ERROR_NOT_FOUND)
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
    ThrowIfFailed(m_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.GetAddressOf())));

    ThrowIfFailed(m_d3d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(m_commandAllocator.GetAddressOf())));

    ThrowIfFailed(m_d3d12Device->CreateCommandList(
        0, // node mask
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr, // initial pipeline state
        IID_PPV_ARGS(m_commandList.GetAddressOf())));

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

dml::Expression Graph::Input(dml::TensorDesc desc)
{
    auto expression = dml::InputTensor(graph, static_cast<uint32_t>(slots.size()), desc);
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

// Append a binding for `desc` at the end of a resource being laid out, aligned
// as the tensor demands, and return the binding with its buffer still unset.
static DML_BUFFER_BINDING AppendBinding(dml::TensorDesc const& desc, uint64_t& resourceSize)
{
    uint32_t alignment = std::max(desc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

    DML_BUFFER_BINDING binding = {};
    binding.Offset = RoundUpToMultiple(resourceSize, static_cast<uint64_t>(alignment));
    binding.SizeInBytes = desc.totalTensorSizeInBytes;

    resourceSize = binding.Offset + binding.SizeInBytes;
    return binding;
}

void Device::CopyArrayToUploadHeap(
    byte* uploadHeapData,
    DML_BUFFER_BINDING const& binding,
    py::array const& array,
    uint32_t index
    )
{
    py::buffer_info info = array.request();
    auto sizeInBytes = static_cast<uint64_t>(info.size) * static_cast<uint64_t>(info.itemsize);

    if ((array.flags() & py::array::c_style) == 0 || sizeInBytes > binding.SizeInBytes)
    {
        throw std::invalid_argument(
            "staged input " + std::to_string(index) +
            " is not a contiguous array that fits its tensor");
    }

    byte* dest = uploadHeapData + binding.Offset;
    memcpy(dest, info.ptr, static_cast<size_t>(sizeInBytes));

    // DirectML rounds a tensor's size up to a 4-byte boundary, so a packed
    // array can come up a few bytes short of it; the tail still has to hold
    // defined values.
    if (sizeInBytes < binding.SizeInBytes)
    {
        memset(dest + sizeInBytes, 0, static_cast<size_t>(binding.SizeInBytes - sizeInBytes));
    }
}

void Device::RecordCopyToBuffer(ID3D12Resource* resource, ID3D12Resource* source, uint64_t sizeInBytes)
{
    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            resource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST)
        );

    m_commandList->CopyBufferRegion(resource, 0, source, 0, sizeInBytes);

    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            resource,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        );
}

void Device::RecordCopyFromBuffer(ID3D12Resource* destination, ID3D12Resource* resource, uint64_t sizeInBytes)
{
    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            resource,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE)
        );

    m_commandList->CopyBufferRegion(destination, 0, resource, 0, sizeInBytes);

    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            resource,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        );
}

DML_BUFFER_BINDING Device::BindBuffer(pydml::Buffer const& buffer, dml::TensorDesc const& desc, uint32_t index)
{
    if (buffer.device.get() != this)
    {
        throw std::invalid_argument(
            "input " + std::to_string(index) + " is a Buffer on a different device");
    }

    if (buffer.SizeInBytes() < desc.totalTensorSizeInBytes)
    {
        throw std::invalid_argument(
            "input " + std::to_string(index) + " is a Buffer of " + std::to_string(buffer.SizeInBytes()) +
            " bytes, short of the " + std::to_string(desc.totalTensorSizeInBytes) + " its tensor needs");
    }

    buffer.resource->UpdateResidency(&m_residencySet);

    DML_BUFFER_BINDING binding = {};
    binding.Buffer = buffer.resource->GetResource();
    binding.Offset = 0;
    binding.SizeInBytes = desc.totalTensorSizeInBytes;
    return binding;
}

void Device::UploadStagedInputs(
    py::iterable staged,
    std::vector<pydml::InputSlot> const& slots,
    std::vector<DML_BUFFER_BINDING> const& bindings,
    gpgmm::d3d12::ResourceAllocation* inputsResource,
    uint64_t inputsResourceSize,
    bool owned
    )
{
    if (inputsResourceSize == 0)
    {
        return;
    }

    // The wrapper layer validated shapes and dtypes and converts one array at a
    // time as this loop pulls on `staged`, so only one converted copy is alive
    // at once. The guards below only keep a bad caller of the private API from
    // corrupting a neighboring tensor's bytes.
    byte* uploadHeapData = nullptr;
    ThrowIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));

    try
    {
        for (py::handle item : staged)
        {
            auto pair = item.cast<std::pair<uint32_t, py::array>>();
            uint32_t index = pair.first;

            if (index >= slots.size() || slots[index].owned != owned)
            {
                throw std::invalid_argument(
                    "staged input " + std::to_string(index) + " is not bound in this phase");
            }

            if (bindings[index].Buffer != inputsResource->GetResource())
            {
                throw std::invalid_argument(
                    "staged input " + std::to_string(index) + " is already bound to a Buffer");
            }

            CopyArrayToUploadHeap(uploadHeapData, bindings[index], pair.second, index);
        }
    }
    catch (...)
    {
        m_uploadHeap->Unmap(0, nullptr);
        throw;
    }

    m_uploadHeap->Unmap(0, nullptr);

    RecordCopyToBuffer(inputsResource->GetResource(), m_uploadHeap->GetResource(), inputsResourceSize);
}

py::list Device::Dispatch(
    pydml::CompiledOperator& model,
    BufferMap const& buffers,
    py::iterable staged,
    bool readback
    )
{
    if (!model.initialized)
    {
        throw std::invalid_argument("the operator must be initialized before it is dispatched");
    }

    IDMLCompiledOperator* op = model.op.Get();

    // Lay out the inputs: owned ones live in the persistent resource and are
    // not bound here, Buffers are bound where they are, and the rest are
    // packed into the device's inputs resource.
    std::vector<DML_BUFFER_BINDING> inputBindings(model.inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < model.inputs.size(); ++i)
    {
        auto& slot = model.inputs[i];
        if (slot.owned)
        {
            continue;
        }

        auto bound = buffers.find(static_cast<uint32_t>(i));
        if (bound != buffers.end())
        {
            inputBindings[i] = BindBuffer(*bound->second, slot.desc, static_cast<uint32_t>(i));
            continue;
        }

        inputBindings[i] = AppendBinding(slot.desc, inputsResourceSize);
    }

    // Lay out the outputs: a resource of its own per output when they stay on
    // the GPU, otherwise packed into the device's outputs resource.
    std::vector<DML_BUFFER_BINDING> outputBindings(model.outputDescs.size());
    std::vector<std::shared_ptr<pydml::Buffer>> outputBuffers;
    uint64_t outputsResourceSize = 0;

    for (size_t i = 0; i < model.outputDescs.size(); ++i)
    {
        auto const& desc = model.outputDescs[i];

        if (readback)
        {
            outputBindings[i] = AppendBinding(desc, outputsResourceSize);
            continue;
        }

        auto resource = CreateDefaultBuffer(
            m_resourceAllocator.Get(),
            RoundUpToMultiple<uint64_t>(desc.totalTensorSizeInBytes, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
        resource->UpdateResidency(&m_residencySet);

        outputBindings[i].Buffer = resource->GetResource();
        outputBindings[i].Offset = 0;
        outputBindings[i].SizeInBytes = desc.totalTensorSizeInBytes;
        outputBuffers.push_back(std::make_shared<pydml::Buffer>(model.device, desc, std::move(resource)));
    }

    DML_BINDING_PROPERTIES bindingProps = op->GetBindingProperties();

    EnsureUploadHeapSize(inputsResourceSize);
    EnsureDefaultBufferSize(inputsResourceSize, m_inputsResource);
    if (readback)
    {
        EnsureReadBackHeapSize(outputsResourceSize);
        EnsureDefaultBufferSize(outputsResourceSize, m_outputsResource);
    }
    EnsureDefaultBufferSize(bindingProps.TemporaryResourceSize, m_temporaryResource);
    EnsureDescriptorHeapSize(bindingProps.RequiredDescriptorCount);

    for (auto& binding : inputBindings)
    {
        if (binding.SizeInBytes != 0 && binding.Buffer == nullptr)
        {
            binding.Buffer = m_inputsResource->GetResource();
        }
    }

    for (auto& binding : outputBindings)
    {
        if (binding.SizeInBytes != 0 && binding.Buffer == nullptr)
        {
            binding.Buffer = m_outputsResource->GetResource();
        }
    }

    assert(model.persistentResource->GetResource()->GetDesc().Width >= bindingProps.PersistentResourceSize);

    UploadStagedInputs(staged, model.inputs, inputBindings, m_inputsResource.Get(), inputsResourceSize, false);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->m_Heap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->m_Heap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    ThrowIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    std::vector<DML_BINDING_DESC> inputBindingDescs(inputBindings.size());
    for (size_t i = 0; i < inputBindings.size(); ++i)
    {
        inputBindingDescs[i] = inputBindings[i].SizeInBytes != 0
            ? DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &inputBindings[i] }
            : DML_BINDING_DESC { DML_BINDING_TYPE_NONE, nullptr };
    }

    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    std::vector<DML_BINDING_DESC> outputBindingDescs(outputBindings.size());
    for (size_t i = 0; i < outputBindings.size(); ++i)
    {
        outputBindingDescs[i] = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &outputBindings[i] };
    }

    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

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

    m_commandList->SetDescriptorHeaps(1, m_descriptorHeap->m_Heap.GetAddressOf());
    m_commandRecorder->RecordDispatch(m_commandList.Get(), op, m_bindingTable.Get());

    if (!readback)
    {
        // The outputs stay on the GPU. A UAV barrier makes them complete
        // before whatever reads them next.
        m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
        ExecuteCommandListAndWait();

        py::list outputs;
        for (auto& buffer : outputBuffers)
        {
            outputs.append(py::cast(buffer));
        }
        return outputs;
    }

    if (outputsResourceSize != 0)
    {
        RecordCopyFromBuffer(m_readbackHeap->GetResource(), m_outputsResource->GetResource(), outputsResourceSize);
    }
    ExecuteCommandListAndWait();

    return DownloadFromReadBackHeap(outputsResourceSize, model.outputDescs, outputBindings);
}

py::array Device::ReadArray(byte const* data, dml::TensorDesc const& desc)
{
    auto const& type = GetDataType(desc.dataType);

    std::vector<py::ssize_t> shape(desc.sizes.begin(), desc.sizes.end());

    // numpy strides are in bytes. An empty vector makes py::array compute
    // packed C-order strides itself.
    std::vector<py::ssize_t> strides;
    if (desc.strides)
    {
        strides.reserve(desc.strides->size());
        for (auto stride : *desc.strides)
        {
            strides.push_back(static_cast<py::ssize_t>(stride) * static_cast<py::ssize_t>(type.itemSize));
        }
    }

    // This constructor copies: it views the mapped memory through the desc's
    // shape and strides, then materializes a packed array that owns its data.
    return py::array(py::dtype(type.numpyName), shape, strides, data);
}

py::list Device::DownloadFromReadBackHeap(
    uint64_t outputsResourceSize,
    std::vector<dml::TensorDesc> const& outputDescs,
    std::vector<DML_BUFFER_BINDING> const& outputBindings
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
            outputs.append(ReadArray(readbackHeapData + outputBindings[i].Offset, outputDescs[i]));
        }

        m_readbackHeap->Unmap(0, nullptr);
    }

    return outputs;
}

std::shared_ptr<pydml::Buffer> Device::Upload(dml::TensorDesc desc, py::array array)
{
    uint64_t sizeInBytes = desc.totalTensorSizeInBytes;

    auto resource = CreateDefaultBuffer(
        m_resourceAllocator.Get(),
        RoundUpToMultiple<uint64_t>(sizeInBytes, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));
    resource->UpdateResidency(&m_residencySet);

    EnsureUploadHeapSize(sizeInBytes);

    DML_BUFFER_BINDING binding = { nullptr, 0, sizeInBytes };

    byte* uploadHeapData = nullptr;
    ThrowIfFailed(m_uploadHeap->Map(0, nullptr, reinterpret_cast<void**>(&uploadHeapData)));
    try
    {
        CopyArrayToUploadHeap(uploadHeapData, binding, array, 0);
    }
    catch (...)
    {
        m_uploadHeap->Unmap(0, nullptr);
        throw;
    }
    m_uploadHeap->Unmap(0, nullptr);

    RecordCopyToBuffer(resource->GetResource(), m_uploadHeap->GetResource(), sizeInBytes);
    ExecuteCommandListAndWait();

    return std::make_shared<pydml::Buffer>(shared_from_this(), std::move(desc), std::move(resource));
}

py::array Device::Download(pydml::Buffer const& buffer)
{
    if (buffer.device.get() != this)
    {
        throw std::invalid_argument("the Buffer belongs to a different device");
    }

    uint64_t sizeInBytes = buffer.SizeInBytes();

    EnsureReadBackHeapSize(sizeInBytes);
    buffer.resource->UpdateResidency(&m_residencySet);

    RecordCopyFromBuffer(m_readbackHeap->GetResource(), buffer.resource->GetResource(), sizeInBytes);
    ExecuteCommandListAndWait();

    CD3DX12_RANGE readRange(0, static_cast<size_t>(sizeInBytes));
    byte* readbackHeapData = nullptr;
    ThrowIfFailed(m_readbackHeap->Map(0, &readRange, reinterpret_cast<void**>(&readbackHeapData)));

    py::array result;
    try
    {
        result = ReadArray(readbackHeapData, buffer.desc);
    }
    catch (...)
    {
        m_readbackHeap->Unmap(0, nullptr);
        throw;
    }
    m_readbackHeap->Unmap(0, nullptr);

    return result;
}

void Device::Initialize(
    pydml::CompiledOperator& model,
    BufferMap const& buffers,
    py::iterable staged
    )
{
    IDMLCompiledOperator* op = model.op.Get();

    ThrowIfFailed(m_operatorInitializer->Reset(1, &op));

    // Only the inputs owned by DirectML are bound at initialize; the initializer
    // takes them as one buffer array with an entry per graph input.
    std::vector<DML_BUFFER_BINDING> inputBindings(model.inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < model.inputs.size(); ++i)
    {
        auto& slot = model.inputs[i];
        if (!slot.owned)
        {
            continue;
        }

        auto bound = buffers.find(static_cast<uint32_t>(i));
        if (bound != buffers.end())
        {
            inputBindings[i] = BindBuffer(*bound->second, slot.desc, static_cast<uint32_t>(i));
            continue;
        }

        inputBindings[i] = AppendBinding(slot.desc, inputsResourceSize);
    }

    uint64_t temporaryResourceSize = m_operatorInitializer->GetBindingProperties().TemporaryResourceSize;
    uint64_t persistentResourceSize = op->GetBindingProperties().PersistentResourceSize;
    uint32_t descriptorHeapSize = m_operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    EnsureUploadHeapSize(inputsResourceSize);
    EnsureDefaultBufferSize(inputsResourceSize, m_inputsResource);
    EnsureDefaultBufferSize(temporaryResourceSize, m_temporaryResource);
    model.allocator = m_resourceAllocator;
    EnsureDefaultBufferSize(persistentResourceSize, model.persistentResource);
    EnsureDescriptorHeapSize(descriptorHeapSize);

    for (auto& binding : inputBindings)
    {
        if (binding.SizeInBytes != 0 && binding.Buffer == nullptr)
        {
            binding.Buffer = m_inputsResource->GetResource();
        }
    }

    UploadStagedInputs(staged, model.inputs, inputBindings, m_inputsResource.Get(), inputsResourceSize, true);

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_operatorInitializer.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->m_Heap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->m_Heap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = descriptorHeapSize;

    ThrowIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    DML_BUFFER_ARRAY_BINDING inputArrayBinding = { static_cast<UINT>(inputBindings.size()), inputBindings.data() };
    DML_BINDING_DESC inputBindingDesc = { DML_BINDING_TYPE_BUFFER_ARRAY, &inputArrayBinding };
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
        ThrowIfFailed(m_d3d12Device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(d3d12DescriptorHeap.GetAddressOf())));

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

#pragma warning(pop)
