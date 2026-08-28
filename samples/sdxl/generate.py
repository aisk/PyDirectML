"""Generate an image from a prompt with Stable Diffusion XL, end to end.

    python generate.py "an astronaut riding a horse on mars"

Four models run in sequence: two CLIP towers turn the prompt into conditioning,
the UNet predicts and removes noise over a few dozen steps, and the VAE decoder
turns the final latent into pixels. Each is built and initialized only when it is
needed and released once it is done, because all of them resident at once do not
comfortably fit.

Sampling is classifier-free guidance, so the UNet runs twice per step: once on
the prompt and once on the negative prompt. The difference between the two is
what gets amplified.
"""

import argparse
import gc
import sys
import time

import numpy as np

import directml as dml

import euler
import unet as unet_module
import vae
from roundtrip import save_image
from text_encoder import TextEncoders
from weights import load_text_encoders, load_tokenizers, load_unet, load_vae

DEFAULT_PROMPT = "an astronaut riding a horse on mars, highly detailed"


def gibibytes(size):
    return f"{size / (1 << 30):.2f} GiB"


def dimensions(text):
    """``1024`` or ``1024x1360``: a width and a height, each a multiple of 8."""
    parts = text.lower().split("x")
    if len(parts) not in (1, 2) or not all(part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"expected WIDTH or WIDTHxHEIGHT, got {text!r}")

    width, height = int(parts[0]), int(parts[-1])
    for value in (width, height):
        if value <= 0 or value % vae.SCALE_FACTOR:
            raise argparse.ArgumentTypeError(
                f"each side must be a positive multiple of {vae.SCALE_FACTOR}")
    return width, height


def encode_prompts(device, prompt, negative, checkpoint=None):
    """Run both CLIP towers over both prompts, then let them go."""
    encoders = TextEncoders(device, load_text_encoders(checkpoint), load_tokenizers())
    conditioning = [encoders.encode(text) for text in (negative, prompt)]
    del encoders
    gc.collect()
    return conditioning


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default="", help="what to steer away from")
    parser.add_argument("--size", type=dimensions, default=(1024, 1024),
                        help="WIDTH or WIDTHxHEIGHT in pixels, each a multiple of 8 "
                             "(default: 1024x1024, what SDXL was trained at). Two "
                             "sides that are each a multiple of 256 need 3.3 GiB "
                             "less than most other sizes; anything else warns")
    parser.add_argument("--sampler", choices=("euler", "euler_a"), default="euler",
                        help="euler_a adds fresh noise back at every step, which is "
                             "what ComfyUI and A1111 default to")
    parser.add_argument("--spacing", choices=("leading", "linspace"), default="leading",
                        help="which timesteps to visit: leading is SDXL's published "
                             "config, linspace is what ComfyUI calls the normal schedule")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="how far to push from the negative prompt towards the prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-batch", action="store_true",
                        help="run both classifier-free guidance passes as one batch "
                             "of two. Measured slower, not faster, and it hangs the "
                             "device from 1024x1024 up; see the README")
    parser.add_argument("--tile-decode", action="store_true",
                        help="decode in overlapping tiles, which cuts the largest "
                             "allocation a run makes down to a fixed cost, at the "
                             "price of an approximation across the seams")
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--checkpoint",
                        help="a single-file LDM checkpoint -- the format ComfyUI "
                             "and A1111 use -- instead of the hub weights")
    args = parser.parse_args()
    width, height = args.size

    # Not an error -- the size runs. It just spends 3.3 GiB more than it has
    # to, and more where a shallower level is off its own alignment as well,
    # which on a 16 GiB card can be the difference between running and hanging
    # the device. Warned here rather than where the UNet is built so it arrives
    # before the fifteen seconds of loading instead of after.
    if unet_module.weights_are_duplicated(height, width):
        alternative = unet_module.nearby_aligned(height, width)
        instead = f" {width}x{alternative} does not." if alternative else ""
        print(f"warning: at {width}x{height} DirectML keeps a second copy of the "
              f"UNet's widest weights, at least 3.3 GiB more than the size "
              f"needs.{instead}")

    device = dml.Device(use_gpu=True)
    started = time.perf_counter()

    print("Encoding the prompt")
    (negative_embeds, negative_pooled), (embeds, pooled) = encode_prompts(
        device, args.prompt, args.negative, args.checkpoint)
    print(f"  {list(embeds.shape)} sequence, {list(pooled.shape)} pooled "
          f"[{time.perf_counter() - started:.0f} s]")

    if args.sampler == "euler_a":
        scheduler = euler.EulerAncestralDiscreteScheduler(seed=args.seed, spacing=args.spacing)
    else:
        scheduler = euler.EulerDiscreteScheduler(spacing=args.spacing)
    scheduler.set_timesteps(args.steps)

    rng = np.random.RandomState(args.seed)
    latents = (rng.randn(1, vae.LATENT_CHANNELS,
                         height // vae.SCALE_FACTOR, width // vae.SCALE_FACTOR)
               * scheduler.init_noise_sigma).astype(np.float32)

    print("Building the UNet")
    params = load_unet(args.checkpoint)
    model = unet_module.UNet(device, params, height, width,
                             batch=2 if args.cfg_batch else 1)
    params.clear()
    gc.collect()
    print(f"  {model.input_count} graph inputs across two graphs, "
          f"{gibibytes(model.persistent_size)} of weights and "
          f"{gibibytes(model.temporary_size)} of scratch "
          f"[{time.perf_counter() - started:.0f} s]")

    # SDXL conditions on the resolution it is pretending to have been cropped
    # from as well as on the prompt, height first.
    resolution = (height, width)
    inputs = [unet_module.conditioning(0, p, resolution, (0, 0), resolution)[1]
              for p in (negative_pooled, pooled)]
    contexts = [e.reshape(1, 1, *e.shape) for e in (negative_embeds, embeds)]

    # Guidance needs the prediction under both conditionings, and they differ
    # only in those two tensors -- the latent and the timestep are the same. So
    # they can go through as one batch of two instead of one after the other.
    if model.batch == 2:
        both_inputs, both_contexts = np.concatenate(inputs), np.concatenate(contexts)

    print(f"Sampling {width}x{height}, {args.steps} steps of {args.sampler}, "
          f"{args.spacing} spacing")
    for step, timestep in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(latents, step)
        time_input, _ = unet_module.conditioning(timestep, pooled, resolution, (0, 0), resolution)

        if model.batch == 2:
            both = model(np.concatenate([model_input] * 2),
                         np.concatenate([time_input] * 2), both_inputs, both_contexts)
            uncond, cond = both[0:1].astype(np.float32), both[1:2].astype(np.float32)
        else:
            uncond, cond = (model(model_input, time_input, add_input, context).astype(np.float32)
                            for add_input, context in zip(inputs, contexts))
        prediction = uncond + args.guidance * (cond - uncond)

        latents = scheduler.step(prediction, step, latents)
        print(f"\r  step {step + 1}/{args.steps}, sigma {scheduler.sigmas[step]:6.3f}"
              f"  [{time.perf_counter() - started:.0f} s]", end="", flush=True)
    print()

    del model
    gc.collect()

    print("Decoding")
    # The decoder holds few weights and enormous intermediates: it runs at the
    # full image size with up to 512 channels, so its scratch is the largest
    # single allocation an aligned run makes, and tiling is what bounds it.
    params = load_vae(args.checkpoint)
    if args.tile_decode:
        decoder = vae.TiledDecoder(device, params, height, width)
        print(f"  {decoder.tiles} tiles of {decoder.tile * vae.SCALE_FACTOR} px, "
              f"{gibibytes(decoder.temporary_size)} of scratch")
    else:
        decoder = vae.decoder(device, params, height, width)
        print(f"  {gibibytes(decoder.temporary_size)} of scratch")
    image, = decoder.run(latents / vae.SCALING_FACTOR)

    print(f"Wrote {save_image(image, args.output)} [{time.perf_counter() - started:.0f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
