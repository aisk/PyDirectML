"""Where the tensors come from: the Hugging Face hub, or one local checkpoint.

diffusers publishes SDXL as a directory of per-model safetensors under the names
these graphs use, and that is what every loader here fetches by default. ComfyUI
and A1111 use a single file holding all three models under LDM's names instead;
pass one to any loader as ``path`` and ``ldm.py`` translates it.

The VAE and the text encoders are built as float32 graphs, so their tensors are
widened on the way out regardless of how the checkpoint stored them. The UNet is
built at half precision and keeps it.
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.numpy import load_file

import ldm

VAE_REPO = "stabilityai/sdxl-vae"
VAE_FILE = "diffusion_pytorch_model.safetensors"

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
TEXT_ENCODERS = ("text_encoder", "text_encoder_2")


def as_float32(array):
    """Widen a checkpoint tensor to float32, including bfloat16."""
    if array.dtype == np.dtype("V2"):
        # safetensors surfaces bfloat16 as an opaque 2-byte void type. bfloat16
        # is the top half of a float32, so widening is a 16-bit shift.
        bits = array.view("<u2").astype("<u4") << 16
        return np.ascontiguousarray(bits.view("<f4").reshape(array.shape))
    return np.ascontiguousarray(array, np.float32)


def read_prefix(path, prefix):
    """Read just the tensors under ``prefix``, with the prefix stripped off.

    Reading one model at a time is what lets ``generate.py`` keep its memory
    discipline against a single-file checkpoint: it loads the text encoders,
    releases them, and only then loads the UNet. The file is 6.5 GiB and never
    has to be resident whole.
    """
    with safe_open(path, framework="np") as checkpoint:
        return {name[len(prefix):]: checkpoint.get_tensor(name)
                for name in checkpoint.keys() if name.startswith(prefix)}


def load_vae(path=None, repo=VAE_REPO, filename=VAE_FILE):
    """The VAE's tensors as float32, from ``path`` or from the hub."""
    if path is not None:
        tensors = ldm.vae(read_prefix(path, "first_stage_model."))
    else:
        tensors = load_file(hf_hub_download(repo_id=repo, filename=filename))
    return {name: as_float32(tensor) for name, tensor in tensors.items()}


def load_text_encoders(path=None, repo=SDXL_REPO):
    """Both CLIP text towers as float32. 3.3 GiB."""
    if path is not None:
        towers = ldm.text_encoders(read_prefix(path, "conditioner.embedders."))
    else:
        towers = {name: load_file(hf_hub_download(repo, f"{name}/model.safetensors"))
                  for name in TEXT_ENCODERS}
    return {name: {k: as_float32(v) for k, v in tower.items()}
            for name, tower in towers.items()}


def load_unet(path=None, repo=SDXL_REPO, half=True):
    """The UNet. 5.1 GiB at half precision, 10.3 at single.

    Half is not an approximation of what SDXL does -- it is what SDXL is run at
    everywhere -- and it is the difference between the weights fitting on a
    16 GiB card alongside their activations and not.
    """
    if path is not None:
        tensors = ldm.unet(read_prefix(path, "model.diffusion_model."))
    else:
        variant = ".fp16" if half else ""
        tensors = load_file(hf_hub_download(
            repo, f"unet/diffusion_pytorch_model{variant}.safetensors"))

    if not half:
        return {name: as_float32(tensor) for name, tensor in tensors.items()}
    return {name: np.ascontiguousarray(tensor, np.float16)
            for name, tensor in tensors.items()}


def load_tokenizers(repo=SDXL_REPO):
    """The two BPE tokenizers, from transformers. No PyTorch involved.

    Always from the hub: a single-file checkpoint carries weights only, and the
    49408-entry vocabulary is not among them.
    """
    from transformers import CLIPTokenizer

    return {name: CLIPTokenizer.from_pretrained(repo, subfolder=folder)
            for name, folder in zip(TEXT_ENCODERS, ("tokenizer", "tokenizer_2"))}


def _report(name, tensors):
    size = sum(t.nbytes for t in tensors.values())
    print(f"  {name:<16} {len(tensors):>5} tensors, {size / (1 << 20):>6.0f} MiB")


def main():
    """``python weights.py`` lists the VAE; give it a checkpoint to translate one."""
    import sys

    if len(sys.argv) < 2:
        tensors = load_vae()
        total = sum(t.nbytes for t in tensors.values())
        print(f"{len(tensors)} tensors, {total / (1 << 20):.0f} MiB as float32")
        for name in sorted(tensors):
            print(f"  {name:<58} {list(tensors[name].shape)}")
        return

    path = sys.argv[1]
    print(path)
    _report("vae", ldm.vae(read_prefix(path, "first_stage_model.")))
    for name, tower in ldm.text_encoders(read_prefix(path, "conditioner.embedders.")).items():
        _report(name, tower)
    _report("unet", ldm.unet(read_prefix(path, "model.diffusion_model.")))


if __name__ == "__main__":
    main()
