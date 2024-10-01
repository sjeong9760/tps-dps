import torch
import numpy as np


class Synthetic:
    def __init__(self, args):
        self.kB = 8.6173303e-5
        self.std = np.sqrt(2 * self.kB * args.temperature * args.timestep)
        self.log_prob = torch.distributions.Normal(0, self.std).log_prob
        self.start_position = torch.tensor([-1.118, 0], dtype=torch.float32).to(
            args.device
        )
        self.target_position = torch.tensor([1.118, 0], dtype=torch.float32).to(
            args.device
        )

    def energy_function(self, position):
        position.requires_grad_(True)
        x = position[:, 0]
        y = position[:, 1]
        term_1 = 4 * (1 - x**2 - y**2) ** 2
        term_2 = 2 * (x**2 - 2) ** 2
        term_3 = ((x + y) ** 2 - 1) ** 2
        term_4 = ((x - y) ** 2 - 1) ** 2
        potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
        force = -torch.autograd.grad(potential.sum(), position)[0]
        position.requires_grad_(False)
        return force, potential.detach()
