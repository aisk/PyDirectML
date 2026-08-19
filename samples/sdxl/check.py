"""Check the DirectML VAE graphs against the NumPy reference.

Runs at a deliberately tiny resolution -- the reference is a pure-NumPy
convolution and would take minutes at 512x512 -- but exercises every layer the
full-size graph uses, including the mid-block attention.
"""

import argparse
import sys

import numpy as np

import directml as dml

import reference
import vae
from weights import load_vae

# float32 accumulated over a 50-layer network in a different order on a different
# device; a few units in the last place per layer add up to about this.
TOLERANCE = 2e-3


def report(name, got, want):
    difference = np.abs(got - want).max()
    scale = np.abs(want).max()
    relative = difference / scale if scale else difference
    status = "ok " if relative < TOLERANCE else "FAIL"
    print(f"  {status} {name:<8} shape {list(got.shape)}  "
          f"max |diff| {difference:.3e}  relative {relative:.3e}")
    return relative < TOLERANCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64,
                        help="image size to test at, a multiple of 8 (default: 64)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size % vae.SCALE_FACTOR:
        parser.error(f"--size must be a multiple of {vae.SCALE_FACTOR}")

    print("Loading weights")
    params = load_vae()
    device = dml.Device(use_gpu=True, use_debug_layer=False)
    rng = np.random.RandomState(args.seed)

    latent_size = args.size // vae.SCALE_FACTOR
    latent = rng.randn(1, vae.LATENT_CHANNELS, latent_size, latent_size).astype(np.float32)
    image = rng.uniform(-1, 1, (1, 3, args.size, args.size)).astype(np.float32)

    print(f"Decoder at {args.size}x{args.size}")
    decoder = vae.decoder(device, params, args.size, args.size)
    print(f"  {decoder.input_count} graph inputs")
    decoded, = decoder.run(latent)
    passed = report("decode", decoded, reference.decode(latent, params))

    print(f"Encoder at {args.size}x{args.size}")
    encoder = vae.encoder(device, params, args.size, args.size)
    print(f"  {encoder.input_count} graph inputs")
    encoded, = encoder.run(image)
    passed &= report("encode", encoded, reference.encode(image, params))

    # Each model owns its persistent resource, which holds the weights DirectML
    # took at initialization. Initializing the encoder must not disturb the
    # decoder's copy -- it did when that buffer belonged to the device.
    print("Decoder again, now that the encoder is initialized too")
    passed &= report("decode", decoder.run(latent)[0], decoded)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
