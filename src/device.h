//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <gpgmm_d3d12.h>

namespace pydml
{
    class Device : public std::enable_shared_from_this<Device>
    {
    public:
        explicit Device(bool useGpu = true, bool useDebugLayer = false, DXGI_GPU_PREFERENCE gpuPreference = DXGI_GPU_PREFERENCE_UNSPECIFIED);

        // Inputs already on the GPU, by input index. They are bound in place;
        // nothing is uploaded for them.
        using BufferMap = std::map<uint32_t, std::shared_ptr<pydml::Buffer>>;

        // Bind the DML_TENSOR_FLAG_OWNED_BY_DML inputs and run the operator
        // initializer, leaving the result in the operator's persistent resource.
        // Once done, those inputs live on the GPU and dispatching does not touch
        // them again. `staged` yields (input index, C-contiguous array of the
        // tensor's own dtype) pairs; the Python wrapper validates and converts,
        // and yielding lazily means only one converted copy exists at a time.
        void Initialize(
            pydml::CompiledOperator& op,
            BufferMap const& buffers,
            py::iterable staged
            );

        // Upload the inputs that are not owned by DirectML, run the operator, and
        // return the outputs: as numpy arrays shaped by the output descs when
        // `readback` is set, otherwise as Buffers that stay on the GPU.
        // Requires an initialized operator.
        py::list Dispatch(
            pydml::CompiledOperator& op,
            BufferMap const& buffers,
            py::iterable staged,
            bool readback
            );

        // Copy a C-contiguous array of the desc's dtype into a new Buffer.
        std::shared_ptr<pydml::Buffer> Upload(dml::TensorDesc desc, py::array array);

        // Read a Buffer back as a numpy array shaped by its desc.
        py::array Download(pydml::Buffer const& buffer);

        inline bool UseGpu() const
        {
            return m_useGpu;
        }

        inline IDMLDevice* GetDevice() const
        {
            return m_dmlDevice.Get();
        }

    protected:
        void RecordOutputReadBack(uint64_t outputsResourceSize);

        py::list DownloadFromReadBackHeap(
            uint64_t outputsResourceSize,
            std::vector<dml::TensorDesc> const& outputDescs,
            std::vector<DmlBufferBinding>& outputBindings
            );

        // A packed numpy array of `desc`'s shape and dtype, copied out of mapped
        // readback memory that holds the tensor in `desc`'s layout.
        py::array ReadArray(byte const* data, dml::TensorDesc const& desc);

        // The binding for input `index` when it is supplied as a Buffer: checks
        // the buffer belongs to this device and is large enough, and marks it
        // resident for the coming command list.
        DmlBufferBinding BindBuffer(pydml::Buffer const& buffer, dml::TensorDesc const& desc, uint32_t index);

        // Copy one array into mapped upload-heap memory at the binding's offset,
        // zero-padding the tail up to the binding's size.
        void CopyArrayToUploadHeap(byte* uploadHeapData, DmlBufferBinding const& binding, py::array const& array, uint32_t index);

        // Record a copy of `sizeInBytes` between two DEFAULT/UPLOAD/READBACK
        // buffers, with `resource` transitioned out of and back into UAV state.
        void RecordCopyToBuffer(ID3D12Resource* resource, ID3D12Resource* source, uint64_t sizeInBytes);
        void RecordCopyFromBuffer(ID3D12Resource* destination, ID3D12Resource* resource, uint64_t sizeInBytes);

        // Copy `staged` (index, array) pairs into the upload heap at the offsets
        // in `bindings`, zero-padding each tensor's tail, and record the copy
        // into `inputsResource`.
        void UploadStagedInputs(
            py::iterable staged,
            std::vector<pydml::InputSlot> const& slots,
            std::vector<DmlBufferBinding> const& bindings,
            gpgmm::d3d12::ResourceAllocation* inputsResource,
            uint64_t inputsResourceSize,
            bool owned
            );

        void EnsureUploadHeapSize(uint64_t requestedSizeInBytes);
        void EnsureReadBackHeapSize(uint64_t requestedSizeInBytes);
        void EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation>& buffer);
        void EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors);

        void ClearGpuBuffers(dml::Span<ID3D12Resource*> buffers);

        void ExecuteCommandListAndWait();

        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocator> m_resourceAllocator;
        
        // Residency management is used to handle oversubscribing of video memory. 
        // The lifetime of |m_residencyManager| will be fully owned by |m_resourceAllocator|.
        gpgmm::d3d12::ResidencyManager* m_residencyManager = nullptr;

        // GPU- and CPU-visible descriptor heaps used for ClearUnorderedAccessView
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapGpu;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapCpu;

        Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_operatorInitializer;
        Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;
        
        // GPU descriptor heaps require explicit residency management since they must
        // stay in a GPU visible memory.
        class SVDescriptorHeap : public gpgmm::d3d12::Heap {
        public:
            SVDescriptorHeap(ComPtr<ID3D12DescriptorHeap> heap, uint64_t size) 
            : gpgmm::d3d12::Heap(heap, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, size), m_Heap(heap) {
            }

            Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_Heap;
        };

        // Lazily-initialized resources for operator initialization/execution
        std::unique_ptr<SVDescriptorHeap> m_descriptorHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_uploadHeap;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_readbackHeap;

        // DEFAULT heap buffers to hold input tensors, output tensors, and temporary resources. The input
        // and output resources are suballocated for operators that have multiple inputs or outputs.
        // The persistent resource is not here: it belongs to the CompiledModel, so that one model's
        // initialized weights survive another model being initialized on the same device.
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_inputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_outputsResource;
        Microsoft::WRL::ComPtr<gpgmm::d3d12::ResourceAllocation> m_temporaryResource;

        gpgmm::d3d12::ResidencySet m_residencySet;

        bool m_useCpuCustomHeapResources = false;
        bool m_useGpu = true;
        DXGI_GPU_PREFERENCE m_gpuPreference;
    };
}