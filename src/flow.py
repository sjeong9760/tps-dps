import torch
import numpy as np
from tqdm import tqdm
from utils.utils import *
from bias import ExternalForce


class FlowNetAgent:
    def __init__(self, args, mds):
        self.policy = ExternalForce(args, mds)
        self.reward = Reward(args, mds)

        if args.training:
            self.replay = ReplayBuffer(args, mds)

    def sample(self, args, mds, temperature):
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        forces = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )

        position, force = mds.report()

        positions[:, 0] = position
        forces[:, 0] = force

        mds.reset()
        mds.set_temperature(temperature)
        for s in tqdm(range(1, args.num_steps + 1), desc="Sampling"):
            bias = (
                self.policy(position.detach(), mds.target_position).squeeze().detach()
            )
            mds.step(bias)

            position, force = mds.report()

            positions[:, s] = position
            forces[:, s] = force - 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
        mds.reset()

        log_reward, final_idx = self.reward(positions, forces)

        if args.training:
            self.replay.add((positions, forces, log_reward))

        for i in range(args.num_samples):
            if final_idx is not None:
                np.save(
                    f"{args.save_dir}/positions/{i}.npy",
                    positions[i][: final_idx[i] + 1].cpu().numpy(),
                )
            else:
                np.save(
                    f"{args.save_dir}/positions/{i}.npy",
                    positions[i].cpu().numpy(),
                )

    def train(self, args, mds):
        optimizer = torch.optim.Adam(
            [
                {"params": [self.policy.log_z], "lr": args.log_z_lr},
                {"params": self.policy.mlp.parameters(), "lr": args.policy_lr},
            ]
        )

        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):

            positions, forces, log_reward = self.replay.sample()

            velocities = (positions[:, 1:] - positions[:, :-1]) / args.timestep

            biases = 1e-6 * self.policy(
                positions.view(-1, positions.size(-2), positions.size(-1)),
                mds.target_position,
            )
            biases = biases.view(*positions.shape)

            means = (
                1 - args.friction * args.timestep
            ) * velocities + args.timestep / mds.m * (forces[:, :-1] + biases[:, :-1])

            log_forward = mds.log_prob(velocities[:, 1:] - means[:, :-1]).mean(
                (1, 2, 3)
            )

            log_z = self.policy.log_z
            tb_loss = (log_z + log_forward - log_reward).square().mean()
            tb_loss.backward()

            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            loss += tb_loss.item()
        loss /= args.trains_per_rollout
        return loss


class ReplayBuffer:
    def __init__(self, args, mds):
        self.positions = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.forces = torch.zeros(
            (args.buffer_size, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.buffer_size = args.buffer_size

    def add(self, data):
        indices = torch.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        (
            self.positions[indices],
            self.forces[indices],
            self.log_reward[indices],
        ) = data

    def sample(self):
        indices = torch.randint(0, min(self.idx, self.buffer_size), (self.batch_size,))
        return (
            self.positions[indices],
            self.forces[indices],
            self.log_reward[indices],
        )


class Reward:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.timestep = args.timestep
        self.friction = args.friction
        self.heavy_atoms = mds.heavy_atoms
        self.target_position = mds.target_position

        self.m = mds.m
        self.log_prob = mds.log_prob

    def __call__(self, positions, forces):
        log_running_reward = self.running_reward(positions, forces)
        log_target_reward, final_idx = self.target_reward(
            positions, self.target_position
        )

        log_reward = log_running_reward + log_target_reward
        return log_reward, final_idx

    def running_reward(self, positions, forces):
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        means = (
            1 - self.friction * self.timestep
        ) * velocities + self.timestep / self.m * forces[:, :-1]
        log_running_reward = self.log_prob(velocities[:, 1:] - means[:, :-1]).mean(
            (1, 2, 3)
        )
        return log_running_reward

    def target_reward(self, positions, target_position):
        positions = positions[:, :, self.heavy_atoms]
        target_position = target_position[:, self.heavy_atoms]
        log_target_reward = torch.zeros(positions.size(0), device=positions.device)
        final_idx = torch.zeros(
            positions.size(0), device=positions.device, dtype=torch.long
        )
        for i in range(positions.size(0)):
            log_target_reward[i], final_idx[i] = self.rmsd(
                positions[i],
                target_position,
            ).max(0)
        return log_target_reward, final_idx

    def rmsd(self, positions, target_position):
        R, t = kabsch(positions, target_position)
        positions = torch.matmul(positions, R.transpose(-2, -1)) + t
        log_target_reward = (
            -0.5 / self.sigma**2 * (positions - target_position).square().mean((-2, -1))
        )
        return log_target_reward
