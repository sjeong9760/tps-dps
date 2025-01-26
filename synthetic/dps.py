import torch
import numpy as np
from tqdm import tqdm
from bias import BiasForce


class DiffusionPathSampler:
    def __init__(self, args, mds):
        self.policy = BiasForce(args)
        self.target_measure = TargetMeasure(args, mds)

        if args.training:
            self.replay = ReplayBuffer(args)

    def sample(self, args, mds, std):
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, 2),
            device=args.device,
        )
        forces = torch.zeros(
            (args.num_samples, args.num_steps + 1, 2),
            device=args.device,
        )
        noises = torch.normal(
            torch.zeros(
                (args.num_samples, args.num_steps, 2),
                device=args.device,
            ),
            torch.ones(
                (args.num_samples, args.num_steps, 2),
                device=args.device,
            ),
        )

        position = mds.start_position.unsqueeze(0)
        force = mds.energy_function(position)[0]
        positions[:, 0] = position
        forces[:, 0] = force

        for s in tqdm(range(args.num_steps), desc="Sampling"):
            bias = (
                self.policy(position.detach(), mds.target_position).squeeze().detach()
            )
            position = position + (force + bias) * args.timestep + std * noises[:, s]
            force = mds.energy_function(position)[0]

            positions[:, s + 1] = position
            forces[:, s + 1] = force

        log_tm, final_idx = self.target_measure(positions, forces)

        if args.training:
            self.replay.add((positions, forces, log_tm))

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

        loss_sum = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):

            positions, forces, log_tm = self.replay.sample()
            biases = self.policy(positions, mds.target_position)
            means = positions + (forces + biases) * args.timestep
            log_bpm = mds.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))

            # Our implementation is based on results in appendix A.2
            log_z = self.policy.log_z
            loss = (log_z + log_bpm - log_tm).square().mean()
            loss.backward()

            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item()
        loss = loss_sum / args.trains_per_rollout
        return loss


class ReplayBuffer:
    def __init__(self, args):
        self.positions = torch.zeros(
            (args.buffer_size, args.num_steps + 1, 2),
            device=args.device,
        )
        self.forces = torch.zeros(
            (args.buffer_size, args.num_steps + 1, 2),
            device=args.device,
        )
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
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


class TargetMeasure:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.log_prob = mds.log_prob
        self.timestep = args.timestep
        self.target_position = mds.target_position

    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri, final_idx = self.relaxed_indicator(positions, self.target_position)

        log_reward = log_upm + log_ri
        return log_reward, final_idx

    def unbiased_path_measure(self, positions, forces):
        means = positions + forces * self.timestep
        log_upm = self.log_prob(positions[:, 1:] - means[:, :-1]).mean((1, 2))
        return log_upm

    def relaxed_indicator(self, positions, target_position):
        log_ri, final_idx = (
            self.rmsd(
                positions.view(-1, positions.size(-1)),
                target_position,
            )
            .view(positions.size(0), positions.size(1))
            .max(1)
        )
        return log_ri, final_idx

    def rmsd(self, positions, target_position):
        msd = (positions - target_position).square().mean(-1)
        log_ri = -0.5 / self.sigma**2 * msd
        return log_ri
