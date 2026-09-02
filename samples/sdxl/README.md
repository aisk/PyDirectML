# SDXL on DirectML

Stable Diffusion XL, built out of the operators `directml` exposes and running
against the real SDXL weights: two CLIP text encoders, the UNet, the VAE, and the
Euler sampler that ties them together.

    python generate.py "an astronaut riding a horse on mars, highly detailed"

63 seconds for a 1024x1024 image at 20 steps on a Radeon RX 6800.

## Running it

```powershell
pip install .[dev] --no-build-isolation   # from the repository root
cd samples\sdxl

python generate.py "a prompt"             # a prompt to an image
python generate.py "a prompt" --size 832x1216 --sampler euler_a --spacing linspace
python generate.py "a prompt" --size 1024x1408 --tile-decode   # a bounded decode
python roundtrip.py                       # encode and decode an image
python encode_prompt.py "a prompt"        # a prompt to the UNet's conditioning
python check.py                           # verify the graphs against NumPy
python euler.py                           # the sampler's schedule and self test
```

Every one of those takes `--checkpoint model.safetensors` to run against a
single-file checkpoint instead of the hub weights.

Weights download from the Hugging Face hub on first use and land in the usual
`~/.cache/huggingface` directory: 320 MiB for the VAE, 3.3 GiB for the two text
encoders, 5.1 GiB for the UNet at half precision. `check.py --part vae` skips
everything but the small one.

Where a 1024x1024 image at 20 steps spends its 63 seconds:

| | |
| --- | --- |
| Encoding the prompt | 7 s, once |
| Building and initializing the UNet | 7 s, once |
| Sampling | 48 s -- 40 UNet forwards at 1.2 s, two per step for guidance |
| Decoding | 1 s, once |

`roundtrip.py` pushes an image through the encoder and back through the decoder.
On a Radeon RX 6800:

| Size | Encode | Decode | Decode, repeated | PSNR |
| --- | --- | --- | --- | --- |
| 512x512 | 0.1 s | 0.3 s | 0.16 s | 34.1 dB |
| 1024x1024 | 0.7 s | 1.0 s | 0.70 s | 36.3 dB |

The first dispatch pays for buffer allocation and meta-command setup, so a
sampling loop runs at the repeated column, not the first one.

That PSNR is the ceiling on any SDXL output: whatever the UNet produces still has
to survive the trip through this decoder.

## What is here

| File | |
| --- | --- |
| `dml_layers.py` | Conv, GroupNorm, LayerNorm, SiLU, attention -- the layers, as DirectML expressions -- and `Model`, which names what feeds which input |
| `vae.py` | The encoder and decoder graphs |
| `text_encoder.py` | Both CLIP towers, and the tokenizing that feeds them |
| `unet.py` | The UNet, as two graphs |
| `euler.py` | `EulerDiscreteScheduler`, in NumPy |
| `ldm.py` | The key names single-file checkpoints use, translated to these |
| `reference.py` | The same VAE and text encoders in NumPy, line for line |
| `check.py` | Runs both and compares |
| `weights.py` | Where the tensors come from: the hub, or one local file |
| `generate.py` | The whole pipeline: a prompt to an image |
| `roundtrip.py`, `encode_prompt.py` | Demos of the VAE and the text encoders on their own |

`reference.py` exists because there is no PyTorch in this repository to check
against. It is slow enough to be obviously correct and fast enough to run at
64x64, which is all it takes to exercise every layer. `check.py` reports about
1e-5 relative error between the two, which is float32 accumulating differently
across fifty layers on two different devices.

## Three operators do most of the work

The bindings expose about 25 operators. Three tricks cover everything the VAE
needs beyond convolution:

**GroupNorm is `mean_variance_normalization` over a regrouped view.** DirectML
normalizes whichever axes you name, so a `[1, C, H, W]` tensor viewed as
`[1, G, C/G, H*W]` and normalized over axes 2 and 3 gets one mean and variance
per group -- exactly GroupNorm. The per-channel affine cannot ride along, because
DirectML wants the scale and bias to be 1 along every normalized axis and the
channel axis is normalized here, so it is a separate multiply and add.

**Broadcasting is a stride of 0.** DirectML reads a tensor through whatever
strides its descriptor carries, and a zero stride makes an axis repeat. A
`[1, C, 1, 1]` bias becomes a `[1, C, H, W]` view with strides `[C, 1, 0, 0]`,
no copy involved. That is what `dml.broadcast()` builds, and nothing broadcasts
without it.

**Transposing is the same trick with non-zero strides.** Attention wants tokens,
not pixels: `[1, C, H, W]` read as `[1, 1, H*W, C]` with strides
`[.., .., 1, H*W]` is the NCHW-to-tokens reshape, again with no copy. So an
attention block is two matrix multiplies around one attention operator, and the
reshapes between them are free.

**And one reshape strides cannot express.** Splitting attention heads is a stride
trick -- `[1, 1, T, C]` read as `[1, heads, T, C/heads]` -- but putting them back
is not. An output channel index decomposes into a head and an offset inside it,
and an index that divides another index is not a stride. So `merge_heads()` views
the buffer as `[1, T, heads, dim]`, which *is* affine, lets an identity operator
write that out packed, and then the last reshape is free. One copy per attention
block, and only in that direction.

That last one is now only on the masked path. Attention was written out here as a
`gemm`, a softmax and a `gemm` until `multihead_attention` was bound, and the head
split and merge existed to feed those; **DirectML has a single operator for the
three, and using it made a 1024x1024 image 107 seconds instead of 63.** See
"One operator for attention" below.

Three operators had to be bound for this sample. `activation_softmax` gained an
`axes` argument: the old binding called `DML_ACTIVATION_SOFTMAX`, which normalizes
a flattened 2-D view and fails outright on the 4-D score matrix attention
produces, where `DML_ACTIVATION_SOFTMAX1` takes axes and does the right thing.
`activation_gelu` is the exact erf form, which is what both the UNet's GEGLU and
OpenCLIP's MLP were trained with. And `multihead_attention` is
`DML_OPERATOR_MULTIHEAD_ATTENTION`.

## The text encoders

SDXL conditions on two CLIP towers at once: ViT-L/14 (768 wide, 12 layers) and
OpenCLIP ViT-bigG/14 (1280 wide, 32 layers). Their per-token outputs concatenate
into the 77x2048 sequence the UNet cross-attends to, and bigG contributes a
pooled 1280-wide vector that joins the timestep embedding.

Three details are easy to get wrong and nothing will complain:

- The embeddings SDXL uses are the **penultimate** layer's output, and the final
  layer norm is not applied to them. It is applied to the last layer's output,
  which is where the pooled vector comes from. So ViT-L only runs 11 of its 12
  layers here -- its pooled output is never used.
- The two towers use different activations. ViT-L was trained with QuickGELU,
  `x * sigmoid(1.702x)`; bigG uses real GELU.
- The two tokenizers pad differently. The first pads with the end-of-text token,
  the second with 0. Pooling finds the real end-of-text either way by taking the
  highest token id, which is what CLIP itself does.

Tokenizing is `transformers.CLIPTokenizer`. BPE over a 49408-entry vocabulary is
string processing, not a DirectML concern, and transformers ships its tokenizers
without needing PyTorch -- nothing in this repository does.

Both towers together are 817 million parameters, 518 graph inputs for bigG alone,
and they encode a prompt in about 290 ms. That happens once per prompt rather
than once per sampling step, so it is not where a generation spends its time.

Checking them turned up a bug in the bindings that had nothing to do with CLIP.
Buffers grew by rounding up to the next power of two, so bigG's 2.78 GiB of
weights asked for a 4 GiB single resource -- and allocating that removed the
device outright, `DXGI_ERROR_DEVICE_REMOVED`, with no other symptom. The
threshold was exactly where the rounding crosses 2 GiB: 2.00 GiB of weights
worked and 2.07 GiB did not. Buffers past 256 MiB now grow by a fixed step
instead.

## The sampler

Diffusion sampling is an ODE solve. Reparameterized by noise level `sigma` the
probability-flow ODE is `dx/dsigma = (x - x0) / sigma`, and a model trained to
predict epsilon returns that right-hand side directly, so one Euler step is one
multiply-add. The content is in the sigma schedule, not the step.

`euler.py` is `diffusers.EulerDiscreteScheduler` under SDXL's published config --
scaled-linear betas, epsilon prediction, linear sigma interpolation -- and
`EulerAncestralDiscreteScheduler` beside it, the `euler_a` that ComfyUI and A1111
default to. The ancestral step integrates *past* the next sigma, down to
`sigma_down`, and then adds fresh noise back up to where it was going, splitting
the two so that `sigma_down^2 + sigma_up^2 == sigma_next^2`. The stochastic
`s_churn` variant is left out.

`--spacing` picks which timesteps to visit. `leading` is SDXL's published config;
`linspace` walks the whole range from 999 down to 0, which is the schedule
ComfyUI calls `normal` -- it starts at the model's true `sigma_max` of 14.61
rather than 11.81 at 27 steps, and finishes at 0.029 rather than 0.041.

**That choice matters much more to `euler_a` than to plain Euler**, and there is
an open problem behind it. One prompt, one seed, 27 steps, `prefectIllustriousXL`:

| | 1024x1024 | 1024x1360 |
| --- | --- | --- |
| `euler`, `leading` | clean | -- |
| `euler`, `linspace` | clean | clean |
| `euler_a`, `leading` | fine black mesh over every flat area | same mesh |
| `euler_a`, `linspace` | clean | same mesh |

Deterministic Euler is clean everywhere it was tried, including 1024x1080,
896x1152 and 896x1160. `euler_a` is not, and `linspace` only rescues it at
1024x1024. **Why is not resolved.** It is not the arithmetic of the ancestral
step -- the oracle test below pins that exactly -- and it is not the non-square
path, because deterministic Euler at the same 1024x1360 comes out clean. ComfyUI
produces a clean image from these settings at this size with its own
`euler_ancestral`, so something else still differs.

The self test comes in three parts. Euler is *exact* on the trajectory the
schedule defines: if a latent really is `clean + sigma * noise` and the model
returns that noise, every step lands exactly on `clean + sigma_next * noise`, and
any sign, ordering or off-by-one error breaks it. That cannot be used on the
ancestral sampler, whose injected noise is not the noise the latent started with,
so the ancestral split is checked against the variance identity, and both
samplers are then run against an *oracle* denoiser -- one handed the clean latent,
which can return the true epsilon for whatever the sample happens to be. Both
have to arrive exactly at the clean latent, and they do, to 1e-8.

`sample(scheduler, denoiser, latents)` takes any callable as the denoiser. The
UNet drops in there.

## The UNet, and the 4 GiB wall

The UNet is 2.57 billion parameters against the text encoders' 817 million and
the VAE's 84 million, and it is the only part that runs more than once per image.
Structurally it is the VAE's encoder and decoder joined at the waist, with two
additions: every resnet has the timestep embedding added to it, which is how one
set of weights behaves differently at different noise levels, and at the two
lower resolutions every resnet is followed by transformer blocks that attend over
the image and then cross-attend to the text. That cross-attention is where the
prompt reaches the pixels.

It runs at half precision, which is what SDXL is run at everywhere. At float32
the weights alone are 10.3 GiB.

**It does not fit in one graph.** DirectML folds a model's `OWNED_BY_DML` weights
into a single persistent buffer, and a single D3D12 buffer stops at 4 GiB --
allocating one past that removes the device (`DXGI_ERROR_DEVICE_REMOVED`) rather
than failing the allocation. The threshold is exact: a graph carrying 3.75 GiB of
weights initializes, one carrying 4.00 GiB does not. The UNet is 4.78 GiB.

So it is two graphs, split at the mid block:

| | |
| --- | --- |
| `conv_in`, both embeddings, three down blocks, mid block | 2.31 GiB |
| three up blocks, `conv_out` | 2.47 GiB |

What crosses between them is the mid-block result, the timestep embedding, and
nine skip connections -- about 54 MiB at 1024x1024. They never leave the GPU:
the first half is dispatched with `readback=False`, which hands back a
`dml.Buffer` per output, and the second half binds those in place. The split is
invisible from outside `unet.UNet`.

`--size` takes a `WIDTHxHEIGHT`, not just a square. A level that halves an odd
extent on the way down cannot get back to it by doubling on the way up, so at
1024x1360 -- latent 128x170, halving to 64x85 and then to 32x43 -- the up path
comes out a pixel taller than the skip connection it has to be concatenated with.
The upsample is trimmed to the next skip's size before its convolution, which is
what diffusers does by handing `Upsample2D` an output size; cropping a
nearest-neighbour 2x upsample to `2n - 1` picks the same source pixel for every
destination as a nearest resize to `2n - 1` would, so nothing is approximated.

Deterministic Euler at 1024x1360, 1024x1080 and 896x1160 -- all of which take
that path -- comes out clean, which is what says the crop is right. What SDXL
does at those sizes is a separate matter: its training buckets are all about a
megapixel, and away from them it starts duplicating the subject, which 896x1160
does.

There is no NumPy reference for the UNet the way there is for the VAE and the
text encoders: 2.57 billion parameters at float32 is 10.3 GiB and minutes per
forward pass. `generate.py` is its test instead. Every part of the pipeline has
to be right for a prompt to come out as a picture of what it asked for -- a wrong
skip order, a transposed attention, the penultimate CLIP layer taken from the
wrong end, and the result is noise rather than a slightly worse image.

## One operator for attention

Attention here was a `gemm`, a softmax and a `gemm`, with the head split and
merge around them. Three operators means the score matrix -- `heads` by tokens by
tokens -- becomes a tensor the graph has to allocate, write, read, write and read
again. DirectML has `DML_OPERATOR_MULTIHEAD_ATTENTION`, which does the three as
one and never hands the score matrix out.

DirectMLX has no helper that emits it, so `src/attention.h` builds the node from
the raw descriptor, in the style of the helpers in `third_party/DirectMLX.h`:
take the input tensor descriptors off the incoming expressions, work out the
output shape, fill in `DML_MULTIHEAD_ATTENTION_OPERATOR_DESC`, and hand the
graph builder an eleven-entry input array of which eight are null -- the stacked
QKV forms, the bias, the mask, the relative position bias and the past key-value
cache are all things this does not use.

Two details are the whole of the work. The operator carries its batch in the
second axis where these graphs carry it in the first, which is the same bytes in
the same order but still has to be said, because both ends of a graph edge have
to agree on the shape; `MoveBatchAxis` reinterprets it across and back, and does
nothing at batch 1. And `DML_MULTIHEAD_ATTENTION_MASK_TYPE` has no additive mask,
while CLIP's causal mask is additive, so `attend()` still writes out the three
operators when it is given one. That is the text encoders only -- every attention
in the UNet and the VAE is unmasked.

It is worth 1.7x on a whole image:

| | before | after |
| --- | --- | --- |
| 1024x1024, 20 steps, end to end | 107 s | **63 s** |
| one UNet forward at 1024x1024 | 2.08 s | 1.18 s |
| down half at 1024x1344, one dispatch | 0.99 s | 0.55 s |
| up half at 1024x1344, one dispatch | 1.25 s | 0.64 s |
| UNet scratch at 1024x1344 | 1.53 GiB | 1.02 GiB |

The numbers agree with the reference implementation to 2.4e-07, which is what the
three operators agree to as well.

**An isolated benchmark of the operator says none of this**, and is worth
recording as a way to be wrong: one level-1 self-attention at 1024x1344 measures
28.1 ms as three operators and 27.5 ms as one, a difference of nothing. Both were
bound by uploading the inputs and reading the output back over PCIe, which the
UNet does once for the whole graph rather than once per attention. Only the
scratch column showed anything -- 1148 MiB against 622 -- and that was the hint
worth following.

## Where the memory goes

`Model.temporary_size` and `Model.persistent_size` report what
`IDMLCompiledOperator::GetBindingProperties` says a compiled graph needs: scratch
for one dispatch, and the bytes DirectML holds between dispatches, which is the
weights in whatever layout the operators wanted them in. `generate.py` prints
both. Against `prefectIllustriousXL`:

| | scratch | weights |
| --- | --- | --- |
| UNet, 1024x1024 | 0.61 GiB | 5.42 GiB |
| UNet, 1024x1360 | 1.05 GiB | **8.75 GiB** |
| UNet, 1024x1408 | 1.09 GiB | 5.42 GiB |
| VAE decoder, 1024x1024 | 5.80 GiB | 0.18 GiB |
| VAE decoder, 1024x1360 | 8.60 GiB | 0.18 GiB |

Both surprises are in that table, and neither is the attention score matrices
that were the obvious suspect -- those live in the scratch column, which stays
around a gigabyte.

**The UNet's weights are not supposed to depend on the image size, and at
1024x1360 there are 3.3 GiB more of them.** DirectML lays a `gemm`'s weight out
for the shape it is about to be multiplied by, and when the row count is not a
multiple of 64 it keeps a second, repacked copy rather than reusing the one it
already has. The rows are tokens. Isolated, one 1280x1280 `gemm` whose weight is
3.125 MiB reports a persistent size of exactly 3.125 MiB for every token count
divisible by 64 of the 96 tried between 64 and 1600, and exactly 6.250 MiB for
all 72 that are not.

Thirty of the UNet's seventy transformer layers are 1280 wide and sit at the
deepest level, where the token count is 1024 at 1024x1024, 1376 at 1024x1360 and
1408 at 1024x1408. Only the middle one is off a multiple of 64, and building the
down half a stage at a time puts the growth exactly where that predicts: nothing
through level 1, +1137 MiB across level 2's twenty transformer layers, +569 MiB
across the mid block's ten. Fifty-seven megabytes per layer, which is the size of
one layer's weights.

So **1024x1408 is the larger image and needs 3.3 GiB less memory than
1024x1360.** `unet.weights_are_duplicated` says whether a size pays for the
second copy, and `generate.py` warns before it loads anything, naming a nearby
size that does not. It is a warning and not an error: the size still runs.

The rule is not "a multiple of 64" -- it is a multiple of 64 *at the deepest
level*, which is a thirty-second of the image. Two sides that are each a multiple
of 256 always satisfy it; most other sizes do not, including six of SDXL's own
nine training buckets:

| | deepest tokens | |
| --- | --- | --- |
| 1024x1024, 1536x640, 640x1536 | 1024, 960 | 5.42 GiB |
| 896x1152, 1152x896, 1344x768, 768x1344 | 1008 | 8.75 GiB |
| 1216x832, 832x1216 | 988 | 8.89 GiB |
| 1280x1280 | 1600 | 5.42 GiB |

1280x1280 has 1.56 times the pixels of 1024x1024 and the same weights; 896x1152
has fewer pixels than either and 3.3 GiB more.

**The VAE decoder's scratch was the largest single allocation an aligned run
made** -- larger, at 1024x1024, than the whole UNet's weights. It runs at the
full image size with up to 512 channels, and every intermediate is a full-size
image. It is not added to the UNet's peak, since `generate.py` releases the UNet
before compiling the decoder, but it was a second peak of the same height.

`--tile-decode` bounds it. The latent is cut into overlapping tiles, one
tile-sized graph is compiled and dispatched over each of them, and the results
are crossfaded back together. What grows with the image is then the number of
dispatches rather than the size of any allocation:

| | scratch | time | |
| --- | --- | --- | --- |
| 1024x1024, whole | 5.80 GiB | 1.0 s | |
| 1024x1024, 9 tiles of 512 px | 1.45 GiB | 1.5 s | 42.98 dB against the whole decode |
| 1024x1344, whole | 8.45 GiB | 2 s | |
| 1024x1344, 9 tiles of 512 px | 1.45 GiB | 2 s | 44.16 dB, mean difference 1.12/255 |

So the peak of an aligned generation stops being the decode and goes back to
being the UNet at 7.0 GiB, and it stops growing with the image at all.

It is an approximation and not a refactoring: GroupNorm normalizes over the whole
feature map, so a tile decoded alone sees different statistics than the same
region does inside a whole-image decode. That is why it is a flag and not the
default. What the crossfade has to earn is that the error stays diffuse rather
than collecting at the seams, and it does -- at 1024x1024 the column-mean
difference at the two tile boundaries is 0.8 and 1.5 standard deviations above
the profile immediately around them, while the largest column difference in the
whole image is at `x=1`, the outer border, where a tile has the least context to
normalize against. The reconstruction PSNR of a round trip goes from 36.25 dB to
36.08.

## Batching guidance, and why it loses

Classifier-free guidance runs the UNet twice per step, once under the negative
conditioning and once under the positive, and the two differ only in two of the
four input tensors. Putting them through as one batch of two is the obvious
optimization, and `--cfg-batch` does it: every graph here already takes its batch
from its input's `.shape`, so it needed only wider placeholders and one
`dml.broadcast`, since a `gemm` will not take a weight whose batch axis disagrees
with its activation's.

It is slower. 768x768, 10 steps, one prompt:

| | per step | weights | scratch |
| --- | --- | --- | --- |
| two dispatches | 1.00 s | 5.42 GiB | 0.27 GiB |
| one batch of two | 1.33 s | 4.78 GiB | 0.49 GiB |

**33% slower.** The GPU is already saturated at batch 1, so batching buys no
parallelism and pays for the wider tensors. It is not the broadcast weight: an
isolated 5376x1280 by 1280x1280 `gemm` measures 10.0 ms at batch 1 against
21.8 ms at batch 2, where twice the batch-1 time would be 19.9, and duplicating
the weight outright instead of broadcasting it measures 22.0 ms, the same.

**It also used to hang the device from 1024x1024 up**, immediately and
reproducibly, on the first dispatch -- and that turned out to be the more
interesting half of the result. Windows resets a GPU whose work has not returned
within `TdrDelay`, two seconds by default and not overridden on this machine.
Batching more than doubles the length of a dispatch:

| 1024x1024, one dispatch | three operators | one operator |
| --- | --- | --- |
| down half, batch 1 | 0.96 s | 0.56 s |
| up half, batch 1 | 1.12 s | 0.62 s |
| down half, batch 2 | `DXGI_ERROR_DEVICE_HUNG` | 1.55 s |
| up half, batch 2 | `DXGI_ERROR_DEVICE_HUNG` | 1.70 s |

The hangs were where doubled halves reached two seconds, and they stopped when
the fused attention brought a half back under a second. No event 4101 is logged
for these hangs, so the ceiling is inferred from the timing rather than confirmed
by the system -- but the timing fits it twice, once going over and once coming
back.

`--cfg-batch` stays because the measurement is worth being able to repeat, and
because `UNet(batch=n)` is the part that would pay off on a GPU that is not
already full. It is off by default.

## Single-file checkpoints

Everything above reads diffusers' layout: a directory with one safetensors file
per model, under the names these graphs use. ComfyUI and A1111 use a single file
holding all three models under the names of Stability's latent-diffusion code,
and `--checkpoint` takes one of those.

    python generate.py "1girl, solo, cherry blossoms" --checkpoint model.safetensors

`ldm.py` is the translation, and it is only renaming -- no graph changes, because
the architecture does not change. Three of the renames are more than a name:
OpenCLIP keeps one fused QKV matrix per layer where HF keeps three, its text
projection is stored transposed, and the VAE's attention projections are 1x1
convolutions where diffusers has plain linears. The rest is index arithmetic. LDM
numbers the down and up paths as one flat list of blocks and stores the decoder's
resolutions in the opposite order, so `input_blocks.7.1.transformer_blocks.3` has
to come back out as `down_blocks.2.attentions.0.transformer_blocks.3`.

Which kind of block a tensor belongs to is read off its own name rather than its
position, because the sub-index that would say so shifts depending on whether the
level has attention at all.

Renaming 2641 keys is the kind of thing that goes wrong quietly, and the UNet has
no NumPy reference to catch it. So `check.py --part layout` compares every
converted name and shape against what diffusers publishes:

```
ok   vae              248 tensors
ok   text_encoder     196 tensors
ok   text_encoder_2   517 tensors
ok   unet            1680 tensors
```

and `check.py --checkpoint` runs the usual numeric comparison on the
checkpoint's own weights. One threshold moves: a finetuned CLIP sits wherever its
training left it, so the pooled-similarity check asks only that a paraphrase beat
an unrelated prompt rather than clearing an absolute bar. Against
`prefectIllustriousXL_v8` that is 0.953 to 0.926, where base SDXL is 0.823 to
0.292 -- the ordering survives, the margin does not.

Two things a checkpoint does not carry. The tokenizers still come from the hub:
the file holds weights only, and the 49408-entry vocabulary is not among them.
And there is no scheduler config in it either, so `euler.py` keeps assuming
epsilon prediction, which is what SDXL and its finetunes are trained with.

## Still missing

- Euler and Euler ancestral only, with two timestep spacings. `s_churn`, Karras
  sigmas, DPM++ and the rest are left out. `euler_a` has the unresolved artifact
  described above.
- Stability at 1024x1360, which looks fixed but is not proven. Runs at that size
  used to die partway through sampling -- five of nine -- with
  `DXGI_ERROR_DEVICE_HUNG`, from a dispatch that had already succeeded several
  times, while 1024x1024 and the in-bucket sizes never did.

  The dispatch-length ceiling above explains it: the up half was 1.25 s there,
  and 1024x1360 is one of the sizes that pays for the duplicated weights, which
  leaves the driver paging on a 16 GiB card -- and a dispatch already most of the
  way to two seconds does not have far to go. That fits the intermittency, which
  neither memory alone nor a fixed limit does. Fusing attention took the halves
  to 0.55 and 0.63 s at that size, and three of three runs came out clean where
  four of nine used to. Three runs is a signal, not a proof.

  A second thing is true at that size and unexplained: the down half's persistent
  buffer is 4.29 GiB, over the 4 GiB line described above, and it initializes and
  dispatches anyway, which that line says it should not.
- The refiner model, and the img2img and inpainting paths.
