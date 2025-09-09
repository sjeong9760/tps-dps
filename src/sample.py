import os
import argparse
import torch

from dynamics.mds import MDs
from dps import DiffusionPathSampler


parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--model_path", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--save_dir", default="paths", type=str)
parser.add_argument("--molecule", default="aldp", type=str)
parser.add_argument("--sim_type", default="1kf1", type=str)

# Policy Config
parser.add_argument("--bias", default="force", type=str)

# Sampling Config
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1, type=float)
parser.add_argument("--sigma", default=0.1, type=float)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)

args = parser.parse_args()

hbond_pairs = [
        # 1st layer
        [
            (56, 249), (251, 444), (446, 639), (641, 54), #1KF1 O-H
            (59, 246), (254, 441), (450, 636), (644, 51), #1KF1 N-H
            (56, 315), (317, 706), (446, 54), (708, 444), #143D O-H
            (60, 312), (320, 703), (449, 51), (712, 441) #143D N-H
        ],
        # 2nd layer
        [
            (89, 282), (284, 477), (479, 672), (674, 87), #1KF1 O-H
            (92, 279), (287, 474), (483, 669), (678, 84), #1KF1 N-H
            (89, 477), (284, 87), (674, 282), #143D O-H (479, 672) is duplicate
            (92, 474), (288, 84), (483, 669), (678, 279) #143D N-H

        ],
        # 3rd layer
        [
            (122, 315), (317, 510), (512, 706), (708, 120), #1KF1 O-H
            (126, 312), (321, 507), (515, 703), (711, 117), #1KF1 N-H
            (122, 249), (251, 639), (512, 120), (641, 510), #143D O-H
            (125, 246), (254, 636), (515, 117), (644, 507) #143D N-H
        ]
    ]

"""
    hbond_pairs = [
        (56, 315), (89, 477), (122, 249), (251, 639), (284, 87), (317, 706), (446, 54), (479, 672), (512, 120), (641, 510), (674, 282), (708, 444), #143D O-H
        (60, 312), (92, 474), (125, 246), (254, 636), (288, 84), (320, 703), (449, 51), (483, 669), (515, 117), (644, 507), (678, 279), (712, 441), #143D N-H
        (56, 249), (89, 282), (122, 315), (251, 444), (284, 477), (317, 510), (446, 639), (479, 672), (512, 706), (641, 54), (674, 87), (708, 120),  #1KF1 O-H
        (59, 246), (92, 279), (126, 312), (254, 441), (287, 474), (321, 507), (450, 636), (483, 669), (515, 703), (644, 51), (678, 84), (711, 117) #1KF1 N-H  
    ]
"""

O6_indices = [54, 87, 120, 249, 282, 315, 444, 477, 510, 639, 672, 706]
cation_indices = [715, 716]

args.cation_indices = cation_indices
args.layer_potential_hbond_pairs = hbond_pairs
args.guanine_o6_indices = O6_indices
args.cation_tolerance_radius = 1  # Example value, adjust as needed
args.h_bond_r0 = 2  # Example value, adjust as needed
args.h_bond_n_min = 3  # Example value, adjust as needed
args.w_cation = 0.1  # Example value, adjust as needed
args.w_hbond = 0.5  # Example value, adjust as needed

if __name__ == "__main__":
    args.training = False
    args.save_dir = f"{args.save_dir}/{args.molecule}/{args.sim_type}/{args.bias}"
    for name in ["positions", "velocities","forces", "biases"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")
    mds = MDs(args)
    agent = DiffusionPathSampler(args, mds)
    agent.policy.load_state_dict(torch.load(args.model_path))
    agent.sample(args, mds, args.temperature)
