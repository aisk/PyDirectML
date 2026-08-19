# SDXL on DirectML

Stable Diffusion XL has four pieces: two text encoders, a UNet, a VAE, and a
sampler that ties them together. This directory has three of the four -- both
text encoders, the VAE, and the sampler -- built out of the operators `directml`
exposes and running against the real SDXL weights.

The UNet is not here yet. See [What is left](#what-is-left).

## Running it

```powershell
pip install .[dev] --no-build-isolation   # from the repository root
cd samples\sdxl

python roundtrip.py                       # encode and decode an image
python encode_prompt.py "a prompt"        # a prompt to the UNet's conditioning
python check.py                           # verify the graphs against NumPy
python euler.py                           # the sampler's schedule and self test
```

Weights download from the Hugging Face hub on first use and land in the usual
`~/.cache/huggingface` directory: 320 MiB for the VAE, 3.3 GiB for the two text
encoders. `check.py --part vae` skips the large download.

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
| `dml_layers.py` | Conv, GroupNorm, LayerNorm, SiLU, attention -- the layers, as DirectML expressions |
| `vae.py` | The encoder and decoder graphs |
| `text_encoder.py` | Both CLIP towers, and the tokenizing that feeds them |
| `euler.py` | `EulerDiscreteScheduler`, in NumPy |
| `reference.py` | The same VAE and text encoders in NumPy, line for line |
| `check.py` | Runs both and compares |
| `weights.py` | Checkpoint download and float32 conversion |
| `roundtrip.py`, `encode_prompt.py` | The demos |

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
no copy involved. That is what `broadcast()` builds.

**Transposing is the same trick with non-zero strides.** Attention wants tokens,
not pixels: `[1, C, H, W]` read as `[1, 1, H*W, C]` with strides
`[.., .., 1, H*W]` is the NCHW-to-tokens reshape, again with no copy. So the
whole attention block is four matrix multiplies and a softmax, and the reshapes
around them are free.

**And one reshape strides cannot express.** Splitting attention heads is a stride
trick -- `[1, 1, T, C]` read as `[1, heads, T, C/heads]` -- but putting them back
is not. An output channel index decomposes into a head and an offset inside it,
and an index that divides another index is not a stride. So `merge_heads()` views
the buffer as `[1, T, heads, dim]`, which *is* affine, lets an identity operator
write that out packed, and then the last reshape is free. One copy per attention
block, and only in that direction.

Two operators had to be bound for this sample. `activation_softmax` gained an
`axes` argument: the old binding called `DML_ACTIVATION_SOFTMAX`, which normalizes
a flattened 2-D view and fails outright on the 4-D score matrix attention
produces, where `DML_ACTIVATION_SOFTMAX1` takes axes and does the right thing.
`activation_gelu` is the exact erf form, which is what both the UNet's GEGLU and
OpenCLIP's MLP were trained with.

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

`euler.py` is `diffusers.EulerDiscreteScheduler` under SDXL's published config:
scaled-linear betas, epsilon prediction, leading timestep spacing, linear sigma
interpolation. Ancestral and stochastic (`s_churn`) variants are left out on
purpose.

Its self test uses the fact that Euler is *exact* on the trajectory the schedule
defines: if a latent really is `clean + sigma * noise` and the model returns that
noise, every step lands exactly on `clean + sigma_next * noise`. Any sign,
ordering, or off-by-one error in the schedule breaks it. Thirty steps land on the
clean latent to 3e-6.

`sample(scheduler, denoiser, latents)` takes any callable as the denoiser. The
UNet drops in there.

## What is left

Only the UNet. It is 2.6 billion parameters against the text encoders' 817
million and the VAE's 84 million, and it is the same building blocks as both plus
cross-attention. Two things in the bindings stood in the way of ever running
something that size; both are fixed now.

**Weights stay on the GPU.** `Device::Compute` used to run the operator
initializer on every call, so nothing could be uploaded once and reused.
Initialization and dispatch are separate now, and the persistent resource -- the
buffer DirectML folds `OWNED_BY_DML` weights into -- belongs to the model rather
than to the device, so several models can be live at once, which is what lets the
two text encoders and the VAE share a device. `compile()` in `dml_layers.py`
initializes; `run()` only dispatches. Repeated 512x512 decoding went from 0.33 s
to 0.16 s, and repeated 1024x1024 from 1.09 s to 0.70 s. The weight-residency
half of that is small at the VAE's 320 MiB; the UNet is 10.3 GiB at float32 and a
20-step run with classifier-free guidance is 40 dispatches, which is where it
stops being a rounding error.

**Half precision works.** A tensor's declared `TensorDataType` is honored end to
end, so float16 weights load as float16 and results come back as float16. The
SDXL UNet is 5.2 GiB at half precision against 10.3 GiB at single, which is the
difference between fitting on a 16 GiB card with room for activations and not.
`samples/dtypes.py` shows both. This sample stays at float32 throughout: the SDXL
VAE is known to overflow in float16, which is why `madebyollin/sdxl-vae-fp16-fix`
exists.

One thing still wants fixing before the UNet: `Binding` keeps a CPU copy of every
weight for the lifetime of the model, even after DirectML has taken the data.
`compile()` drops the arrays the graph was built from, which is 2.8 GiB back for
bigG, but the Binding copies need a change on the C++ side. Fine at 320 MiB, not
fine at 10.3 GiB.
