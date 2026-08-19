# SDXL on DirectML

Stable Diffusion XL has four pieces: two text encoders, a UNet, a VAE, and a
sampler that ties them together. This directory has the VAE and the sampler,
built out of the operators `directml` exposes, running against the real
`stabilityai/sdxl-vae` weights.

The UNet is not here yet. See [What the UNet needs](#what-the-unet-needs).

## Running it

```powershell
pip install .[dev] --no-build-isolation   # from the repository root
cd samples\sdxl

python roundtrip.py                       # encode and decode an image
python check.py                           # verify the graphs against NumPy
python euler.py                           # the sampler's schedule and self test
```

The weights (320 MiB) download from the Hugging Face hub on first use and land in
the usual `~/.cache/huggingface` directory.

`roundtrip.py` pushes an image through the encoder and back through the decoder.
On a Radeon RX 6800:

| Size | Encode | Decode | PSNR |
| --- | --- | --- | --- |
| 512x512 | 0.6 s | 0.5 s | 34.1 dB |
| 1024x1024 | 4.6 s | 9.8 s | 36.3 dB |

That PSNR is the ceiling on any SDXL output: whatever the UNet produces still has
to survive the trip through this decoder.

## What is here

| File | |
| --- | --- |
| `dml_layers.py` | Conv, GroupNorm, SiLU, attention -- the layers, as DirectML expressions |
| `vae.py` | The encoder and decoder graphs |
| `euler.py` | `EulerDiscreteScheduler`, in NumPy |
| `reference.py` | The same VAE in NumPy, line for line |
| `check.py` | Runs both and compares |
| `weights.py` | Checkpoint download and float32 conversion |
| `roundtrip.py` | The demo |

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

The one binding change this sample needed was `activation_softmax`, which now
takes an `axes` argument. The old binding called `DML_ACTIVATION_SOFTMAX`, which
normalizes a flattened 2-D view and fails outright on the 4-D score matrix
attention produces; `DML_ACTIVATION_SOFTMAX1` takes axes and does the right
thing.

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

## What the UNet needs

The UNet is 2.6 billion parameters against the VAE's 84 million, and two things
in the bindings have to change before it is worth writing.

**Weights are re-uploaded on every dispatch.** `Device::Compute` calls
`InitializeOperator` and `DispatchOperator` together on each call
(`src/device.cpp:141`), with a comment conceding the point. For the VAE that is
320 MiB once and does not matter. A 20-step run with classifier-free guidance is
40 UNet dispatches, so at float32 it is 10.3 GiB crossing PCIe forty times.
Splitting initialization from dispatch, so `DML_TENSOR_FLAG_OWNED_BY_DML` weights
stay resident, is the prerequisite.

**Everything is float32.** `Binding` takes a `py::array_t<float>`
(`src/module.cpp:172`) and `TensorData` hands results back as float32
(`src/model.h:41`), so a float16 checkpoint is widened on load. The SDXL UNet is
10.3 GiB at float32 against 5.2 GiB at float16, and 16 GiB cards have to hold
activations too. This is also `docs/api-design.md` §3.2, which wants the dtype
honored for correctness reasons independent of SDXL.

Beyond that the UNet needs the two text encoders (CLIP ViT-L and OpenCLIP
ViT-bigG, 817 million parameters between them) and a BPE tokenizer. Its
cross-attention and timestep embeddings are expressible with what is bound
today; its GEGLU feed-forward is not, and neither is the text encoders' GELU.
Both want `dml::ActivationGelu` or `dml::Erf`, which DirectMLX has
(`third_party/DirectMLX.h:1658`, `:1958`) and `module.cpp` does not bind yet.
