"""Fetch an SDXL checkpoint from the Hugging Face hub and read it as float32.

The DirectML bindings upload float32 -- ``Binding`` takes a ``py::array_t<float>``
and ``TensorData`` hands results back as float32 -- so every tensor is widened on
the way out regardless of how the checkpoint stored it.
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

VAE_REPO = "stabilityai/sdxl-vae"
VAE_FILE = "diffusion_pytorch_model.safetensors"


def as_float32(array):
    """Widen a checkpoint tensor to float32, including bfloat16."""
    if array.dtype == np.dtype("V2"):
        # safetensors surfaces bfloat16 as an opaque 2-byte void type. bfloat16
        # is the top half of a float32, so widening is a 16-bit shift.
        bits = array.view("<u2").astype("<u4") << 16
        return np.ascontiguousarray(bits.view("<f4").reshape(array.shape))
    return np.ascontiguousarray(array, np.float32)


def load_vae(repo=VAE_REPO, filename=VAE_FILE):
    """Download (or reuse the cached) SDXL VAE and return its tensors as float32."""
    path = hf_hub_download(repo_id=repo, filename=filename)
    return {name: as_float32(tensor) for name, tensor in load_file(path).items()}


def main():
    tensors = load_vae()
    total = sum(t.nbytes for t in tensors.values())
    print(f"{len(tensors)} tensors, {total / (1 << 20):.0f} MiB as float32")
    for name in sorted(tensors):
        print(f"  {name:<58} {list(tensors[name].shape)}")


if __name__ == "__main__":
    main()
