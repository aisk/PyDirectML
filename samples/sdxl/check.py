"""Check the DirectML graphs against the NumPy reference.

    python check.py                 # everything
    python check.py --part vae      # just the VAE, which needs 320 MiB of weights
    python check.py --part text     # just the text encoders, which need 3.3 GiB

The VAE runs at a deliberately tiny resolution -- the reference is a pure-NumPy
convolution and would take minutes at 512x512 -- but exercises every layer the
full-size graph uses, including the mid-block attention.

The text encoders get a second check that the NumPy comparison cannot give:
paraphrases of a prompt should land near each other and unrelated prompts should
not. Both implementations agreeing proves they were written the same way; the
similarities coming out sane proves they were written the right way.
"""

import argparse
import sys

import numpy as np

import directml as dml

import reference
import text_encoder
import vae
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


def check_vae(device, size, seed):
    print("Loading VAE weights")
    params = load_vae()
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


def check_text(device):
    print("Loading text encoder weights (3.3 GiB)")
    weights = load_text_encoders()
    tokenizers = load_tokenizers()
    passed = True

    for name, config in text_encoder.CONFIGS.items():
        print(f"{name}: {config['layers']} layers, {config['width']} wide")
        ids = np.array(tokenizers[name](PROMPTS[0], padding="max_length",
                                        max_length=text_encoder.MAX_TOKENS,
                                        truncation=True)["input_ids"], np.uint32)

        model = text_encoder.text_encoder(device, weights[name], config)
        print(f"  {model.input_count} graph inputs")
        outputs = model.run(ids.reshape(1, 1, 1, text_encoder.MAX_TOKENS))
        expected = reference.encode_text(ids.astype(np.int64), weights[name], config)

        for got, want, label in zip(outputs, expected, ("penultimate", "final")):
            passed &= report(label, got, want)

    print("Pooled similarity, which the NumPy comparison cannot vouch for")
    encoders = text_encoder.TextEncoders(device, weights, tokenizers)
    pooled = np.stack([encoders.encode(p)[1] for p in PROMPTS])
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)

    paraphrase, unrelated = pooled[0] @ pooled[1], pooled[0] @ pooled[2]
    sane = paraphrase > 0.6 > unrelated
    print(f"  {'ok  ' if sane else 'FAIL'} paraphrase {paraphrase:+.3f}, "
          f"unrelated {unrelated:+.3f}")
    return passed and sane


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("all", "vae", "text"), default="all")
    parser.add_argument("--size", type=int, default=64,
                        help="image size to check the VAE at (default: 64)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size % vae.SCALE_FACTOR:
        parser.error(f"--size must be a multiple of {vae.SCALE_FACTOR}")

    device = dml.Device(use_gpu=True, use_debug_layer=False)
    passed = True
    if args.part in ("all", "vae"):
        passed &= check_vae(device, args.size, args.seed)
    if args.part in ("all", "text"):
        passed &= check_text(device)

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
