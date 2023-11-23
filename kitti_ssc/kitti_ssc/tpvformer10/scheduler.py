from timm.scheduler import CosineLRScheduler
import torch

class MyCosineLRScheduler(CosineLRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, t_initial: int, t_mul: float = 1, lr_min: float = 0, decay_rate: float = 1, warmup_t=0, warmup_lr_init=0, warmup_prefix=False, cycle_limit=0, t_in_epochs=True, noise_range_t=None, noise_pct=0.67, noise_std=1, noise_seed=42, initialize=True) -> None:
        super().__init__(optimizer, t_initial, t_mul, lr_min, decay_rate, warmup_t, warmup_lr_init, warmup_prefix, cycle_limit, t_in_epochs, noise_range_t, noise_pct, noise_std, noise_seed, initialize)
        self.count = 0

    def step(self, ):
        self.step_update(self.count)
        self.count += 1
        return