import torch
import numpy as np
from tqdm import tqdm

from utils.utils import kabsch
from bias import BiasForce


class DiffusionPathSampler:
    def __init__(self, args, mds):
        self.policy = BiasForce(args, mds)
        self.target_measure = TargetPathMeasure(args, mds)
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
        forces[:, 0] = force
        biases[:, 0] = self.policy(position.detach(), mds.target_position).squeeze().detach()
        velocity = mds.report_vel()
        velocities[:, 0] = velocity
        mds.reset()
        mds.set_temperature(temperature)
        for s in tqdm(range(1, args.num_steps + 1), desc="Sampling"):
            bias = self.policy(position.detach(), mds.target_position).squeeze().detach()
            mds.step(bias)
            position, force = mds.report()
            positions[:, s] = position
            forces[:, s] = force - 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
            velocity = mds.report_vel()
            velocities[:, s] = velocity
            biases[:, s] = bias
        mds.reset()
        log_tpm, final_idx = self.target_measure(positions, forces)
        if args.training:
            self.replay.add((positions, forces, log_tpm))
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

    def train(self, args, mds):
        optimizer = torch.optim.Adam(
            [
                {"params": [self.policy.log_z], "lr": args.log_z_lr},
                {"params": self.policy.mlp.parameters(), "lr": args.policy_lr},
            ]
        )
        loss_sum = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            positions, forces, log_tpm = self.replay.sample()
            velocities = (positions[:, 1:] - positions[:, :-1]) / args.timestep
            biases = 1e-6 * self.policy(
                positions.view(-1, positions.size(-2), positions.size(-1)),
                mds.target_position,
            )
            biases = biases.view(*positions.shape)
            means = (
                1 - args.friction * args.timestep
            ) * velocities + args.timestep / mds.m * (forces[:, :-1] + biases[:, :-1])
            log_bpm = mds.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2, 3))
            # Our implementation is based on results in appendix A.2
            if args.control_variate == "global":
                log_z = self.policy.log_z
            elif args.control_variate == "local":
                log_z = (log_tpm - log_bpm).mean().detach()
            elif args.control_variate == "zero":
                log_z = 0
            loss = (log_z + log_bpm - log_tpm).square().mean()
            loss.backward()
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        loss = loss_sum / args.trains_per_rollout
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
        self.log_tpm = torch.zeros(args.buffer_size, device=args.device)
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
            self.log_tpm[indices],
        ) = data

    def sample(self):
        indices = torch.randint(0, min(self.idx, self.buffer_size), (self.batch_size,))
        return (
            self.positions[indices],
            self.forces[indices],
            self.log_tpm[indices],
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

        ### Penalties for G4
        self.w_cation = args.w_cation
        self.w_hbond = args.w_hbond
        self.cation_indices = torch.tensor(args.cation_indices, device=self.device, dtype=torch.long)
        self.guanine_o6_indices = torch.tensor(args.guanine_o6_indices, device=self.device, dtype=torch.long)
        
        self.layer_potential_hbond_pairs = [
            torch.tensor(pairs, device=self.device, dtype=torch.long)
            for pairs in args.layer_potential_hbond_pairs
        ]
        self.cation_tolerance_radius = args.cation_tolerance_radius
        self.h_bond_r0 = args.h_bond_r0
        self.h_bond_n_min = args.h_bond_n_min


    def __call__(self, positions, forces):
        log_upm = self.unbiased_path_measure(positions, forces)
        log_ri, final_idx = self.relaxed_indicator(positions, self.target_position)

        ## Penalties for G4
        #cation_loss = self.cation_penalty(positions)
        hbond_loss = self.hbond_penalty(positions)

        #log_tpm = log_upm + log_ri - self.w_cation*cation_loss - self.w_hbond*hbond_loss
        log_tpm = log_upm + log_ri - self.w_hbond*hbond_loss

        return log_tpm, final_idx

    def unbiased_path_measure(self, positions, forces):
        velocities = (positions[:, 1:] - positions[:, :-1]) / self.timestep
        means = (
            1 - self.friction * self.timestep
        ) * velocities + self.timestep / self.m * forces[:, :-1]
        log_upm = self.log_prob(velocities[:, 1:] - means[:, :-1]).mean((1, 2, 3))
        return log_upm
    
    def cation_penalty(self, positions):
        guanosine_coords = positions[:, :, self.guanine_o6_indices, :].view(-1, len(self.guanine_o6_indices), 3)
        centroid = torch.mean(guanosine_coords, dim=1, keepdim=True)
        _, _, Vh = torch.linalg.svd(guanosine_coords - centroid)
        axis_vector = Vh[:, 0, :].unsqueeze(1)
        
        cations_pos = positions[:, :, self.cation_indices, :].view(-1, len(self.cation_indices), 3)
        total_penalty = 0.0
        for i in range(len(self.cation_indices)):
            cation_pos = cations_pos[:, i, :].unsqueeze(1)
            
            # 2. 중심축과의 수직 거리 d 계산
            vec_p_c = cation_pos - centroid
            proj_on_axis = torch.sum(vec_p_c * axis_vector, dim=-1, keepdim=True) * axis_vector
            perpendicular_vec = vec_p_c - proj_on_axis
            distances = torch.sqrt(torch.sum(perpendicular_vec.square(), dim=-1)) # shape: (batch*steps, 1)

            # !! 3. "안전 실린더" 페널티 적용 !!
            # 거리가 허용 반경을 초과할 때만 페널티를 부과합니다.
            penalty = torch.nn.functional.relu(distances - self.cation_tolerance_radius).square().mean()
            total_penalty += penalty

        return total_penalty

    def hbond_penalty(self,positions):
        r0 = self.h_bond_r0
        n, m = 8, 12  # 스위칭 함수 파라미터
        n_min_per_layer = self.h_bond_n_min
        epsilon = 1e-6 # 0으로 나누는 것을 방지
        delta = 1e-3
        total_hbond_penalty = 0.0

        all_pairs = torch.cat(self.layer_potential_hbond_pairs, dim=0)
        donors    = positions[:, :, all_pairs[:, 0], :]
        acceptors = positions[:, :, all_pairs[:, 1], :]
        distances = torch.sqrt(torch.sum((donors - acceptors).square(), dim=-1) + epsilon)
        r_ratio_n = (distances / r0).pow(n)
        r_ratio_m = (distances / r0).pow(m)
        switching_values = (1.0 - r_ratio_n) / (1.0 - r_ratio_m + epsilon)
        x = distances  / r0 
        near = (torch.abs(x - 1.0) < delta)
        if near.any():
            switching_values = torch.where(near, torch.full_like(switching_values, float(n)/float(m)), switching_values)
        s = switching_values.clamp(0.0, 1.0) 
        N_eff_total = s.sum(dim=-1) 
        deficit = n_min_per_layer - N_eff_total
        penalty = torch.nn.functional.softplus(deficit).square().mean()
        return penalty


        """ #layer by layer 방식
        for layer_pairs in self.layer_potential_hbond_pairs:
            donors = positions[:, :, layer_pairs[:, 0], :]
            acceptors = positions[:, :, layer_pairs[:, 1], :]
            
            distances = torch.sqrt(torch.sum((donors - acceptors).square(), dim=-1) + epsilon)
            
            # 스위칭 함수 계산 (모든 쌍에 대해 병렬적으로)
            r_ratio_n = (distances / r0).pow(n)
            r_ratio_m = (distances / r0).pow(m)
            switching_values = (1.0 - r_ratio_n) / (1.0 - r_ratio_m + epsilon)
            
            # 각 경로, 각 스텝 별로 유효 결합 수를 합산
            N_current_layer = torch.sum(switching_values, dim=-1)
            
            # 최소 결합 수(N_min)보다 현재 유효 결합 수가 적을 경우에만 페널티 부과
            layer_penalty = torch.nn.functional.relu(n_min_per_layer - N_current_layer).square().mean()
            total_hbond_penalty += layer_penalty
        """
        return total_hbond_penalty


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
