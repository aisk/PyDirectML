"""Euler sampling for SDXL, in NumPy.

Diffusion sampling is an ODE solve. Reparameterized by noise level ``sigma`` the
probability-flow ODE is ``dx/dsigma = (x - x0(x, sigma)) / sigma``, and the model
predicts exactly that right-hand side when it is trained to predict epsilon. So
one Euler step is one multiply-add, and the sampler's real content is the
schedule of sigmas it steps through.

This is ``diffusers.EulerDiscreteScheduler`` under SDXL's published config:
scaled-linear betas, epsilon prediction, leading timestep spacing, linear sigma
interpolation. Ancestral and stochastic (``s_churn``) variants are deliberately
left out.
"""

import numpy as np

NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.012
STEPS_OFFSET = 1


class EulerDiscreteScheduler:
    """The deterministic Euler sampler, indexed by step number."""

    def __init__(self, num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                 beta_start=BETA_START, beta_end=BETA_END, steps_offset=STEPS_OFFSET):
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset

        # "scaled_linear": the betas are linear in sqrt-space, not in beta.
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                            num_train_timesteps, dtype=np.float64) ** 2
        alphas_cumprod = np.cumprod(1.0 - betas)

        # The noise level a sample carries at each training timestep, in the
        # variance-exploding parameterization the sampler works in.
        self.train_sigmas = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)

        self.set_timesteps(50)

    def set_timesteps(self, num_inference_steps):
        """Pick the timesteps to visit, and the sigma at each one."""
        # "leading" spacing: evenly spaced from 0, walked backwards, shifted by
        # steps_offset so the last step lands on t=1 rather than t=0.
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
        """
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


def main():
    scheduler = EulerDiscreteScheduler()
    for steps in (20, 30, 50):
        scheduler.set_timesteps(steps)
        head = ", ".join(f"{s:.3f}" for s in scheduler.sigmas[:4])
        print(f"{steps:>3} steps: init_noise_sigma {scheduler.init_noise_sigma:.4f}  "
              f"timesteps {scheduler.timesteps[0]:.0f}..{scheduler.timesteps[-1]:.0f}  "
              f"sigmas {head}, ... , {scheduler.sigmas[-2]:.4f}, 0")

    print("Self test")
    _self_test()
    print("  ok")


if __name__ == "__main__":
    main()
