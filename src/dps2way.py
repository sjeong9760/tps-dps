import torch
import numpy as np
from tqdm import tqdm

from utils.utils import kabsch
from bias import BiasForce


class DiffusionPathSampler:
    def __init__(self, args, mds):
        self.policyA = BiasForce(args, mds)
        self.policyB = BiasForce(args, mds)

        self.target_measure = TargetPathMeasure(args, mds)

    def sample(self, args, mds, temperature):
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        forces = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        velocities = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        biases = torch.zeros(
            (args.num_samples, args.num_steps + 1, mds.num_particles, 3),
            device=args.device,
        )
        
        position, force = mds.report()
        positions[:, 0] = position
        self.policy = self.policyA

        forces[:, 0] = force
        biases[:, 0] = self.policy(position.detach(), mds.target_position).squeeze().detach()
        velocity = mds.report_vel()
        velocities[:, 0] = velocity
        mds.reset()
        mds.set_temperature(temperature)
        target_position = mds.target_position
        for s in tqdm(range(1, args.num_steps + 1), desc="Sampling"):
            if s % 2 == 0:
                bias = self.policy(position.detach(), target_position).squeeze().detach()
            else:
                bias = torch.zeros(biases[:,0].shape,device=args.device,)
            mds.step(bias)
            position, force = mds.report()
            positions[:, s] = position
            forces[:, s] = force - 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
            velocity = mds.report_vel()
            velocities[:, s] = velocity
            biases[:, s] = bias
            if np.random.rand() < 0.6:
                self.policy = self.policyA
                target_position = mds.target_position
            else:
                self.policy = self.policyB
                target_position = mds.start_position
        mds.reset()
        if args.calc_final:
            _, final_idx = self.target_measure(positions, forces)
        else:
            final_idx = torch.full((args.num_samples,), args.num_steps, device=args.device)
        for i in range(args.num_samples):
            np.save(
                f"{args.save_dir}/positions/{i}.npy",
                positions[i][: final_idx[i] + 1].cpu().numpy(),
            )
            np.save(
                f"{args.save_dir}/velocities/{i}.npy",
                velocities[i][: final_idx[i] + 1].cpu().numpy(),
            )
            np.save(
                f"{args.save_dir}/forces/{i}.npy",
                forces[i][: final_idx[i] + 1].cpu().numpy(),
            )
            np.save(
                f"{args.save_dir}/biases/{i}.npy",
                biases[i][: final_idx[i] + 1].cpu().numpy(),
            )


class TargetPathMeasure:
    def __init__(self, args, mds):
        self.sigma = args.sigma
        self.timestep = args.timestep
        self.friction = args.friction
        self.heavy_atoms = mds.heavy_atoms
        self.target_position = mds.target_position
        self.m = mds.m
        self.log_prob = mds.log_prob
        self.device = args.device

    def __call__(self, positions, forces):
        log_ri, final_idx = self.relaxed_indicator(positions, self.target_position)
        return log_ri, final_idx

    def relaxed_indicator(self, positions, target_position):
        positions = positions[:, :, self.heavy_atoms]
        target_position = target_position[:, self.heavy_atoms]
        log_ri = torch.zeros(positions.size(0), device=positions.device)
        final_idx = torch.zeros(
            positions.size(0), device=positions.device, dtype=torch.long
        )
        for i in range(positions.size(0)):
            log_ri[i], final_idx[i] = self.rbf(
                positions[i],
                target_position,
            ).max(0)
        return log_ri, final_idx

    def rbf(self, positions, target_position):
        R, t = kabsch(positions, target_position)
        positions = torch.matmul(positions, R.transpose(-2, -1)) + t
        log_ri = (
            -0.5 / self.sigma**2 * (positions - target_position).square().mean((-2, -1))
        )
        return log_ri
