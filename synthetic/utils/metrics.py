import torch
import numpy as np


class Metric:
    def __init__(self, args, mds):
        self.std = mds.std
        self.device = args.device
        self.log_prob = mds.log_prob
        self.timestep = args.timestep
        self.save_dir = args.save_dir
        self.num_samples = args.num_samples
        self.energy_function = mds.energy_function
        self.target_position = mds.target_position

    def __call__(self):
        positions, forces, potentials = [], [], []
        for i in range(self.num_samples):
            position = np.load(f"{self.save_dir}/positions/{i}.npy")
            position = torch.from_numpy(position).to(self.device)
            force, potential = self.energy_function(position)
            positions.append(position)
            forces.append(force)
            potentials.append(potential)

        final_position = torch.stack([position[-1] for position in positions])
        rmsd, rmsd_std = self.rmsd(final_position, self.target_position)
        thp, hit = self.thp(final_position, self.target_position)
        etp, etp_std = self.etp(hit, potentials)

        metrics = {
            "rmsd": rmsd,
            "thp": 100 * thp,
            "etp": etp,
            "rmsd_std": rmsd_std,
            "etp_std": etp_std,
        }

        return metrics

    def rmsd(self, final_position, target_position):
        rmsd = (final_position - target_position).square().sum(-1).sqrt()
        mean_rmsd, std_rmsd = rmsd.mean().item(), rmsd.std().item()
        return mean_rmsd, std_rmsd

    def thp(self, final_position, target_position):
        hit = (final_position - target_position).square().sum(-1) < 0.5**2
        thp = hit.sum().float() / len(hit)
        return thp.item(), hit

    def etp(self, hit, potentials):
        etps = []
        for i, hit_idx in enumerate(hit):
            if hit_idx:
                etp = potentials[i].max(0)[0]
                etps.append(etp)

        if len(etps) > 0:
            etps = torch.tensor(etps)
            etp, std_etp = etps.mean().item(), etps.std().item()
            return etp, std_etp
        else:
            return None, None
