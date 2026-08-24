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
    parser.add_argument("--size", type=int, default=1024,
                        help="square size, a multiple of 8 (default: 1024, what SDXL was trained at)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="how far to push from the negative prompt towards the prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--checkpoint",
                        help="a single-file LDM checkpoint -- the format ComfyUI "
                             "and A1111 use -- instead of the hub weights")
    args = parser.parse_args()

    if args.size % vae.SCALE_FACTOR:
        parser.error(f"--size must be a multiple of {vae.SCALE_FACTOR}")

    device = dml.Device(use_gpu=True)
    started = time.perf_counter()

    print("Encoding the prompt")
    (negative_embeds, negative_pooled), (embeds, pooled) = encode_prompts(
        device, args.prompt, args.negative, args.checkpoint)
    print(f"  {list(embeds.shape)} sequence, {list(pooled.shape)} pooled "
          f"[{time.perf_counter() - started:.0f} s]")

    scheduler = euler.EulerDiscreteScheduler()
    scheduler.set_timesteps(args.steps)

    latent_size = args.size // vae.SCALE_FACTOR
    rng = np.random.RandomState(args.seed)
    latents = (rng.randn(1, vae.LATENT_CHANNELS, latent_size, latent_size)
               * scheduler.init_noise_sigma).astype(np.float32)

    print("Building the UNet")
    params = load_unet(args.checkpoint)
    model = unet_module.UNet(device, params, args.size, args.size)
    params.clear()
    gc.collect()
    print(f"  {model.input_count} graph inputs across two graphs "
          f"[{time.perf_counter() - started:.0f} s]")

    # SDXL conditions on the resolution it is pretending to have been cropped
    # from as well as on the prompt.
    resolution = (args.size, args.size)
    inputs = [unet_module.conditioning(0, p, resolution, (0, 0), resolution)[1]
              for p in (negative_pooled, pooled)]
    contexts = [e.reshape(1, 1, *e.shape) for e in (negative_embeds, embeds)]

    print(f"Sampling, {args.steps} steps")
    for step, timestep in enumerate(scheduler.timesteps):
        model_input = scheduler.scale_model_input(latents, step)
        time_input, _ = unet_module.conditioning(timestep, pooled, resolution, (0, 0), resolution)

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
    decoder = vae.decoder(device, load_vae(args.checkpoint), args.size, args.size)
    image, = decoder.run(latents / vae.SCALING_FACTOR)

    print(f"Wrote {save_image(image, args.output)} [{time.perf_counter() - started:.0f} s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
