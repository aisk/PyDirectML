"""Turn a prompt into the conditioning SDXL's UNet takes.

    python encode_prompt.py "an astronaut riding a horse on mars"

Both CLIP towers run on the GPU. What comes out is a 77x2048 sequence, which the
UNet cross-attends to, and a single 1280-wide pooled vector, which joins the
timestep embedding. Encoding happens once per prompt rather than once per
sampling step, so this is not where a generation spends its time -- but it is
where a generation stops being about the prompt and starts being about tensors.
"""

import argparse
import sys
import time

import numpy as np

import directml as dml

from text_encoder import CONFIGS, TextEncoders
from weights import load_text_encoders, load_tokenizers

DEFAULT_PROMPT = "an astronaut riding a horse on mars, highly detailed"


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--compare", help="a second prompt, to compare the pooled vectors")
    parser.add_argument("--save", help="write the conditioning to this .npz")
    parser.add_argument("--checkpoint",
                        help="a single-file LDM checkpoint -- the format ComfyUI "
                             "and A1111 use -- instead of the hub weights")
    args = parser.parse_args()

    device = dml.Device()

    print("Loading weights (3.3 GiB as float32)")
    weights = load_text_encoders(args.checkpoint)
    for name, config in CONFIGS.items():
        total = sum(v.size for v in weights[name].values())
        print(f"  {name:<16} {total / 1e6:5.0f}M params, "
              f"{config['layers']} layers, {config['width']} wide, {config['heads']} heads")

    start = time.perf_counter()
    encoders = TextEncoders(device, weights, load_tokenizers())
    print(f"Built and initialized both towers in {time.perf_counter() - start:.1f} s")
    weights.clear()  # DirectML has the weights now

    start = time.perf_counter()
    embeds, pooled = encoders.encode(args.prompt)
    elapsed = time.perf_counter() - start

    print(f"\n{args.prompt!r}")
    print(f"  encoded in {elapsed * 1000:.0f} ms")
    print(f"  sequence {list(embeds.shape)}  mean {embeds.mean():+.4f}  std {embeds.std():.4f}")
    print(f"  pooled   {list(pooled.shape)}  mean {pooled.mean():+.4f}  std {pooled.std():.4f}")

    if args.compare:
        other_embeds, other_pooled = encoders.encode(args.compare)
        print(f"\n{args.compare!r}")
        print(f"  pooled cosine similarity with the first prompt: {cosine(pooled, other_pooled):+.4f}")
        print(f"  sequence cosine similarity: "
              f"{cosine(embeds.ravel(), other_embeds.ravel()):+.4f}")

    if args.save:
        np.savez(args.save, prompt_embeds=embeds, pooled_prompt_embeds=pooled)
        print(f"\nWrote {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
