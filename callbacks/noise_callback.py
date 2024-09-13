from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

class ExponentialNoiseScheduleCallback(BaseCallback):
    def __init__(self, start_noise: float, end_noise: float, total_steps: int, verbose: int = 0):
        super(ExponentialNoiseScheduleCallback, self).__init__(verbose)
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.total_steps = total_steps
        self.current_step = 0

    def _on_step(self) -> bool:
        # Calculate the exponentially decaying noise scale
        noise_scale = self.start_noise * (self.end_noise / self.start_noise) ** (self.current_step / self.total_steps)

        # Update the noise in the action space
        if isinstance(self.model.action_noise, NormalActionNoise):
            self.model.action_noise.sigma = np.ones(self.model.action_space.shape) * noise_scale
        
        self.current_step += 1
        return True  # Continue training

