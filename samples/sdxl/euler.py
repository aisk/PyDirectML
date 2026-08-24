"""Euler sampling for SDXL, in NumPy.

Diffusion sampling is an ODE solve. Reparameterized by noise level ``sigma`` the
probability-flow ODE is ``dx/dsigma = (x - x0(x, sigma)) / sigma``, and the model
predicts exactly that right-hand side when it is trained to predict epsilon. So
one Euler step is one multiply-add, and the sampler's real content is the
schedule of sigmas it steps through.

This is ``diffusers.EulerDiscreteScheduler`` under SDXL's published config:
scaled-linear betas, epsilon prediction, leading timestep spacing, linear sigma
interpolation -- and ``EulerAncestralDiscreteScheduler`` beside it, which is the
``euler_a`` that ComfyUI and A1111 default to. The stochastic ``s_churn`` variant
is deliberately left out.
"""

import numpy as np

NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.012
STEPS_OFFSET = 1


class EulerDiscreteScheduler:
    """The deterministic Euler sampler, indexed by step number."""

    def __init__(self, num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                 beta_start=BETA_START, beta_end=BETA_END, steps_offset=STEPS_OFFSET,
                 spacing="leading"):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.spacing = spacing

        # "scaled_linear": the betas are linear in sqrt-space, not in beta.
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                            num_train_timesteps, dtype=np.float64) ** 2
        alphas_cumprod = np.cumprod(1.0 - betas)

        # The noise level a sample carries at each training timestep, in the
        # variance-exploding parameterization the sampler works in.
        self.train_sigmas = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)

        self.set_timesteps(50)

    def set_timesteps(self, num_inference_steps):
        """Pick the timesteps to visit, and the sigma at each one.

        "leading" is what SDXL's published config asks for: evenly spaced from 0,
        walked backwards, shifted by ``steps_offset`` so the last step lands on
        t=1 rather than t=0. "linspace" walks the whole range from the last
        training timestep down to 0, which is the schedule ComfyUI and A1111 call
        "normal" -- it starts at a higher noise level and finishes at a lower one.
        """
        if self.spacing == "linspace":
            self.timesteps = np.linspace(self.num_train_timesteps - 1, 0,
                                         num_inference_steps, dtype=np.float64)
        else:
            stride = self.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(num_inference_steps) * stride)[::-1].astype(np.float64)
            self.timesteps = timesteps + self.steps_offset

        sigmas = np.interp(self.timesteps, np.arange(self.num_train_timesteps),
                           self.train_sigmas)
        # A trailing zero so the final step integrates all the way to clean data.
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        return self.timesteps

    @property
    def num_inference_steps(self):
        return len(self.timesteps)

    @property
    def init_noise_sigma(self):
        """The standard deviation to draw the starting latent with.

        Under leading spacing the model input is divided by ``sqrt(sigma^2 + 1)``
        rather than by ``sigma``, so the starting latent is scaled to match.
        Under linspace the first sigma is the largest the model was trained on,
        and the latent is simply drawn at it.
        """
        if self.spacing == "linspace":
            return float(self.sigmas.max())
        return float((self.sigmas.max() ** 2 + 1.0) ** 0.5)

    def scale_model_input(self, sample, step):
        """Normalize a latent to the unit-variance input the model expects."""
        return sample / np.sqrt(self.sigmas[step] ** 2 + 1.0)

    def add_noise(self, sample, noise, step):
        """Bring a clean latent up to the noise level of ``step``."""
        return sample + noise * self.sigmas[step]

    def step(self, model_output, step, sample):
        """One Euler step, from the noise level of ``step`` to the next one."""
        sigma = self.sigmas[step]
        # With epsilon prediction the derivative is the model output itself; it
        # is spelled out here because that is only true for this prediction type.
        predicted_original = sample - sigma * model_output
        derivative = (sample - predicted_original) / sigma
        return sample + derivative * (self.sigmas[step + 1] - sigma)


class EulerAncestralDiscreteScheduler(EulerDiscreteScheduler):
    """Euler, with the noise put back that the step just took out.

    A deterministic step integrates straight from one sigma to the next. The
    ancestral step integrates *past* it, down to ``sigma_down``, and then adds
    fresh noise back up to where it was going -- so the sample keeps arriving at
    the right noise level, but by a different route each time.

    The split is the one that preserves variance,
    ``sigma_down^2 + sigma_up^2 == sigma_next^2``, with ``sigma_up`` taken as
    large as that identity allows. Sampling is no longer a function of the
    starting latent alone, which is why the noise has its own seed.
    """

    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState(seed)

    def noise_split(self, step):
        """How far this step integrates, and how much noise it puts back."""
        sigma, sigma_next = float(self.sigmas[step]), float(self.sigmas[step + 1])
        up = sigma_next * np.sqrt(sigma ** 2 - sigma_next ** 2) / sigma
        return float(np.sqrt(max(sigma_next ** 2 - up ** 2, 0.0))), float(up)

    def step(self, model_output, step, sample):
        """One Euler step to ``sigma_down``, then noise back up to ``sigma_next``."""
        sigma = self.sigmas[step]
        down, up = self.noise_split(step)

        predicted_original = sample - sigma * model_output
        derivative = (sample - predicted_original) / sigma
        stepped = sample + derivative * (down - sigma)

        noise = self.rng.randn(*sample.shape).astype(sample.dtype)
        return stepped + noise * up


def sample(scheduler, denoiser, latents):
    """Run the full sampling loop.

    ``denoiser(model_input, timestep)`` returns the predicted noise. The UNet
    goes here; the sampler does not care what produces the prediction.
    """
    for step, timestep in enumerate(scheduler.timesteps):
        prediction = denoiser(scheduler.scale_model_input(latents, step), timestep)
        latents = scheduler.step(prediction, step, latents)
    return latents


def _self_test():
    """Euler is exact on the trajectory the schedule itself defines.

    If a latent really is ``clean + sigma * noise`` and the model returns that
    noise, then every Euler step lands exactly on ``clean + sigma_next * noise``.
    Any sign, ordering or off-by-one error in the schedule breaks this.
    """
    rng = np.random.RandomState(0)
    clean = rng.randn(1, 4, 8, 8).astype(np.float32)
    noise = rng.randn(1, 4, 8, 8).astype(np.float32)

    scheduler = EulerDiscreteScheduler()
    scheduler.set_timesteps(30)

    latents = scheduler.add_noise(clean, noise, 0)
    worst = 0.0
    for step, _ in enumerate(scheduler.timesteps):
        latents = scheduler.step(noise, step, latents)
        expected = clean + noise * scheduler.sigmas[step + 1]
        worst = max(worst, float(np.abs(latents - expected).max()))

    print(f"  exact-trajectory error over 30 steps: {worst:.3e}")
    assert worst < 1e-4, "Euler step does not integrate the schedule exactly"
    assert np.all(np.diff(scheduler.sigmas) < 0), "sigmas must decrease"
    assert scheduler.sigmas[-1] == 0.0, "the schedule must end at zero noise"
    assert np.abs(latents - clean).max() < 1e-4, "sampling did not reach the clean latent"


def _ancestral_self_test():
    """The ancestral split has to leave the sample at the sigma it was heading for.

    Integrating to ``sigma_down`` and adding independent noise of scale
    ``sigma_up`` composes to noise of scale ``sqrt(down^2 + up^2)``, so that has
    to come out as ``sigma_next`` or the trajectory drifts off the schedule.
    """
    scheduler = EulerAncestralDiscreteScheduler()
    scheduler.set_timesteps(30)

    worst = 0.0
    for step in range(scheduler.num_inference_steps):
        down, up = scheduler.noise_split(step)
        worst = max(worst, abs((down ** 2 + up ** 2) ** 0.5 - scheduler.sigmas[step + 1]))
        assert 0.0 <= down <= scheduler.sigmas[step + 1], "sigma_down is out of range"

    print(f"  ancestral variance error over 30 steps: {worst:.3e}")
    assert worst < 1e-5, "the ancestral split does not preserve the noise level"
    assert scheduler.noise_split(scheduler.num_inference_steps - 1) == (0.0, 0.0),         "the last step must add no noise"


def _oracle_test():
    """Both samplers land on the clean latent when the denoiser is perfect.

    The exact-trajectory test cannot be used on the ancestral sampler, because
    the noise it injects is not the noise the latent started with. But a denoiser
    that is handed the clean latent can return the true epsilon for whatever the
    sample happens to be, and against that oracle both samplers have to arrive
    exactly at the clean latent -- which pins the ancestral step's arithmetic the
    same way, and catches a mis-scaled ``sigma_up`` that the variance identity
    alone would not.
    """
    rng = np.random.RandomState(0)
    clean = rng.randn(1, 4, 8, 8).astype(np.float32)
    noise = rng.randn(1, 4, 8, 8).astype(np.float32)

    for scheduler in (EulerDiscreteScheduler(), EulerAncestralDiscreteScheduler()):
        scheduler.set_timesteps(30)
        latents = scheduler.add_noise(clean, noise, 0)

        for step in range(scheduler.num_inference_steps):
            epsilon = (latents - clean) / scheduler.sigmas[step]
            latents = scheduler.step(epsilon, step, latents)

        error = float(np.abs(latents - clean).max())
        print(f"  {type(scheduler).__name__:<32} against an oracle: {error:.3e}")
        assert error < 1e-4, "the sampler does not reach the clean latent"


def main():
    for spacing in ("leading", "linspace"):
        scheduler = EulerDiscreteScheduler(spacing=spacing)
        for steps in (20, 30, 50):
            scheduler.set_timesteps(steps)
            head = ", ".join(f"{s:.3f}" for s in scheduler.sigmas[:3])
            print(f"{spacing:>8}, {steps:>3} steps: init_noise_sigma "
                  f"{scheduler.init_noise_sigma:7.4f}  timesteps "
                  f"{scheduler.timesteps[0]:.0f}..{scheduler.timesteps[-1]:.0f}  "
                  f"sigmas {head}, ... , {scheduler.sigmas[-2]:.4f}, 0")

    print("Self test")
    _self_test()
    _ancestral_self_test()
    _oracle_test()
    print("  ok")


if __name__ == "__main__":
    main()
