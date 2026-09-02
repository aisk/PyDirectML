//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

using Microsoft::WRL::ComPtr;

void WaitForQueueToComplete(ID3D12CommandQueue* queue)
{
    ComPtr<ID3D12Device> device;
    ThrowIfFailed(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
    ComPtr<ID3D12Fence> fence;
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())), device.Get());
    ThrowIfFailed(queue->Signal(fence.Get(), 1), device.Get());
    ThrowIfFailed(fence->SetEventOnCompletion(1, nullptr), device.Get());

    // Every dispatch drains the queue through here, so this is where a removal
    // gets attributed to the work that actually caused it rather than to
    // whichever unrelated call notices first.
    ThrowIfDeviceRemoved(device.Get());
}
