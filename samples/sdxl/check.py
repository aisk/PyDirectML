"""Check the DirectML graphs against the NumPy reference.

    python check.py                 # everything
    python check.py --part vae      # just the VAE, which needs 320 MiB of weights
    python check.py --part text     # just the text encoders, which need 3.3 GiB
    python check.py --checkpoint model.safetensors   # against a translated checkpoint

The VAE runs at a deliberately tiny resolution -- the reference is a pure-NumPy
convolution and would take minutes at 512x512 -- but exercises every layer the
full-size graph uses, including the mid-block attention.

The text encoders get a second check that the NumPy comparison cannot give:
paraphrases of a prompt should land near each other and unrelated prompts should
not. Both implementations agreeing proves they were written the same way; the
similarities coming out sane proves they were written the right way.

``--part layout`` is for single-file checkpoints. There is no NumPy reference for
the UNet, and its 1680 renamed keys are exactly where a mapping table goes wrong,
so instead it compares what ``ldm.py`` produces against the names and shapes
diffusers publishes.
"""

import argparse
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import directml as dml

import ldm
import reference
import text_encoder
import vae
import weights
from weights import load_text_encoders, load_tokenizers, load_vae

# float32 accumulated over dozens of layers in a different order on a different
# device; a few units in the last place per layer add up to about this.
TOLERANCE = 2e-3

PROMPTS = [
    "an astronaut riding a horse on mars",
    "a photograph of an astronaut on a horse",
    "a bowl of ramen on a wooden table",
]


def report(name, got, want):
    got = got.reshape(want.shape)
    difference = np.abs(got - want).max()
    scale = np.abs(want).max()
    relative = difference / scale if scale else difference
    passed = relative < TOLERANCE
    print(f"  {'ok  ' if passed else 'FAIL'} {name:<12} shape {list(got.shape)}  "
          f"max |diff| {difference:.3e}  relative {relative:.3e}")
    return passed


def check_layout(path):
    """Compare a translated checkpoint against the layout diffusers publishes.

    Only the safetensors headers are read on the reference side, but the files
    have to be on disk -- 8.6 GiB of them, which is why this is not part of a
    plain ``check.py`` run.
    """
    def reference_shapes(repo, filename):
        with safe_open(hf_hub_download(repo, filename), framework="np") as f:
            return {name: tuple(f.get_slice(name).get_shape()) for name in f.keys()}

    towers = ldm.text_encoders(weights.read_prefix(path, "conditioner.embedders."))
    converted = {
        "vae": ldm.vae(weights.read_prefix(path, "first_stage_model.")),
        "text_encoder": towers["text_encoder"],
        "text_encoder_2": towers["text_encoder_2"],
        "unet": ldm.unet(weights.read_prefix(path, "model.diffusion_model.")),
    }
    published = {
        "vae": (weights.VAE_REPO, weights.VAE_FILE),
        "text_encoder": (weights.SDXL_REPO, "text_encoder/model.safetensors"),
        "text_encoder_2": (weights.SDXL_REPO, "text_encoder_2/model.safetensors"),
        "unet": (weights.SDXL_REPO, "unet/diffusion_pytorch_model.fp16.safetensors"),
    }

    passed = True
    for name, tensors in converted.items():
        got = {key: tuple(tensor.shape) for key, tensor in tensors.items()}
        want = reference_shapes(*published[name])

        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        mismatched = sorted(key for key in set(got) & set(want) if got[key] != want[key])
        ok = not (missing or extra or mismatched)
        passed &= ok

        print(f"  {'ok  ' if ok else 'FAIL'} {name:<16} {len(got)} tensors")
        for key in missing[:5]:
            print(f"         missing {key} {want[key]}")
        for key in extra[:5]:
            print(f"         extra   {key} {got[key]}")
        for key in mismatched[:5]:
            print(f"         shape   {key} got {got[key]}, want {want[key]}")
    return passed


def check_vae(device, size, seed, checkpoint=None):
    print("Loading VAE weights")
    params = load_vae(checkpoint)
    rng = np.random.RandomState(seed)

    latent_size = size // vae.SCALE_FACTOR
    latent = rng.randn(1, vae.LATENT_CHANNELS, latent_size, latent_size).astype(np.float32)
    image = rng.uniform(-1, 1, (1, 3, size, size)).astype(np.float32)

    print(f"Decoder at {size}x{size}")
    decoder = vae.decoder(device, params, size, size)
    print(f"  {decoder.input_count} graph inputs")
    decoded, = decoder.run(latent)
    passed = report("decode", decoded, reference.decode(latent, params))

    print(f"Encoder at {size}x{size}")
    encoder = vae.encoder(device, params, size, size)
    print(f"  {encoder.input_count} graph inputs")
    encoded, = encoder.run(image)
    passed &= report("encode", encoded, reference.encode(image, params))

    # Each model owns its persistent resource, which holds the weights DirectML
    # took at initialization. Initializing the encoder must not disturb the
    # decoder's copy -- it did when that buffer belonged to the device.
    print("Decoder again, now that the encoder is initialized too")
    passed &= report("decode", decoder.run(latent)[0], decoded)
    return passed


def check_text(device, checkpoint=None):
    print("Loading text encoder weights (3.3 GiB)")
    params = load_text_encoders(checkpoint)
    tokenizers = load_tokenizers()
    passed = True

    for name, config in text_encoder.CONFIGS.items():
        print(f"{name}: {config['layers']} layers, {config['width']} wide")
        ids = np.array(tokenizers[name](PROMPTS[0], padding="max_length",
                                        max_length=text_encoder.MAX_TOKENS,
                                        truncation=True)["input_ids"], np.uint32)

        model = text_encoder.text_encoder(device, params[name], config)
        print(f"  {model.input_count} graph inputs")
        outputs = model.run(ids.reshape(1, 1, 1, text_encoder.MAX_TOKENS))
        expected = reference.encode_text(ids.astype(np.int64), params[name], config)

        for got, want, label in zip(outputs, expected, ("penultimate", "final")):
            passed &= report(label, got, want)

    print("Pooled similarity, which the NumPy comparison cannot vouch for")
    encoders = text_encoder.TextEncoders(device, params, tokenizers)
    pooled = np.stack([encoders.encode(p)[1] for p in PROMPTS])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)

    paraphrase, unrelated = pooled[0] @ pooled[1], pooled[0] @ pooled[2]
    if checkpoint is None:
        sane = paraphrase > 0.6 > unrelated
    else:
        # A finetuned CLIP sits wherever its training left it, and a checkpoint
        # trained on tags rather than sentences can put every English pair above
        # 0.9 of each other. Only the ordering is still worth asserting.
        sane = paraphrase > unrelated
    print(f"  {'ok  ' if sane else 'FAIL'} paraphrase {paraphrase:+.3f}, "
          f"unrelated {unrelated:+.3f}")
    return passed and sane


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("all", "vae", "text", "layout"), default="all")
    parser.add_argument("--checkpoint",
                        help="a single-file LDM checkpoint to check instead of the hub weights")
    parser.add_argument("--size", type=int, default=64,
                        help="image size to check the VAE at (default: 64)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size % vae.SCALE_FACTOR:
        parser.error(f"--size must be a multiple of {vae.SCALE_FACTOR}")

    if args.part == "layout":
        if not args.checkpoint:
            parser.error("--part layout needs a --checkpoint to compare")
        print(f"Layout of {args.checkpoint} against the diffusers weights")
        passed = check_layout(args.checkpoint)
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1

    device = dml.Device(use_gpu=True, use_debug_layer=False)
    passed = True
    if args.part in ("all", "vae"):
        passed &= check_vae(device, args.size, args.seed, args.checkpoint)
    if args.part in ("all", "text"):
        passed &= check_text(device, args.checkpoint)

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
