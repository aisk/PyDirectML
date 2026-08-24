"""Run an image through the SDXL VAE on the GPU and back.

    python roundtrip.py [image] [--size 512]

Encoding then decoding is the honest way to see the autoencoder work end to end:
the latent it produces is the same space SDXL's UNet diffuses in, and the decoded
image is exactly what the last step of a text-to-image run would emit. What comes
back is close to the input but not identical -- the VAE throws away detail on its
way down to a 64x64x4 latent, and that loss is the floor on any SDXL output.
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

import directml as dml

import vae
from weights import load_vae

SAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_image(path, size):
    image = ImageOps.fit(Image.open(path).convert("RGB"), (size, size), Image.BICUBIC)
    array = np.asarray(image, np.float32) / 255.0
    # HWC in [0, 1] to NCHW in [-1, 1], which is what the encoder was trained on.
    return array.transpose(2, 0, 1)[None] * 2.0 - 1.0


def save_image(array, path):
    array = np.clip(array[0].transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0)
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8)).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=os.path.join(SAMPLES_DIR, "DefaultImage.jpg"))
    parser.add_argument("--size", type=int, default=512,
                        help="square size to work at, a multiple of 8 (default: 512)")
    parser.add_argument("--output", default="roundtrip.png")
    parser.add_argument("--latents", help="also write the latent to this .npy file")
    parser.add_argument("--checkpoint",
                        help="a single-file LDM checkpoint -- the format ComfyUI "
                             "and A1111 use -- instead of the hub weights")
    args = parser.parse_args()

    if args.size % vae.SCALE_FACTOR:
        parser.error(f"--size must be a multiple of {vae.SCALE_FACTOR}")
    if not os.path.exists(args.image):
        parser.error(f"no such image: {args.image}")

    original = load_image(args.image, args.size)
    params = load_vae(args.checkpoint)
    device = dml.Device(use_gpu=True, use_debug_layer=False)
    print(f"{device}")

    latent_size = args.size // vae.SCALE_FACTOR
    print(f"Encoding {args.size}x{args.size} to {latent_size}x{latent_size}x{vae.LATENT_CHANNELS}")
    encoder = vae.encoder(device, params, args.size, args.size)
    start = time.perf_counter()
    latent, = encoder.run(original)
    print(f"  {time.perf_counter() - start:.1f} s")

    # The UNet works on scaled latents; unscaling is the decoder's first act.
    scaled = latent * vae.SCALING_FACTOR
    print(f"  latent mean {scaled.mean():+.3f}, std {scaled.std():.3f}")
    if args.latents:
        np.save(args.latents, scaled)
        print(f"  wrote {args.latents}")

    print("Decoding")
    decoder = vae.decoder(device, params, args.size, args.size)
    start = time.perf_counter()
    decoded, = decoder.run(scaled / vae.SCALING_FACTOR)
    print(f"  {time.perf_counter() - start:.1f} s")

    a = np.clip(original * 0.5 + 0.5, 0.0, 1.0)
    b = np.clip(decoded * 0.5 + 0.5, 0.0, 1.0)
    psnr = 10.0 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))
    print(f"Reconstruction PSNR {psnr:.2f} dB")
    print(f"Wrote {save_image(decoded, args.output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
