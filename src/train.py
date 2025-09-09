import os
import argparse

import torch
import wandb

from dynamics.mds import MDs
from utils.logging import Logger
from dps import DiffusionPathSampler


def main():
    parser = argparse.ArgumentParser()
    # System Config
    parser.add_argument("--date", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--molecule", default="aldp", type=str)
    parser.add_argument('--wandb', action='store_true', default=False)
    # Logger Config
    parser.add_argument("--save_dir", default="results", type=str)
    # Policy Config
    parser.add_argument("--bias", default="force", type=str)
    # Sampling Config
    parser.add_argument("--start_state", default="c5", type=str)
    parser.add_argument("--end_state", default="c7ax", type=str)
    parser.add_argument("--num_steps", default=1000, type=int)
    parser.add_argument("--timestep", default=1, type=float)
    parser.add_argument("--sigma", default=0.1, type=float)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--temperature", default=300, type=float)
    parser.add_argument("--friction", default=0.001, type=float)
    # Training Config
    parser.add_argument("--start_temperature", default=600, type=float)
    parser.add_argument("--end_temperature", default=300, type=float)
    parser.add_argument("--num_rollouts", default=1000, type=int)
    parser.add_argument("--trains_per_rollout", default=1000, type=int)
    parser.add_argument("--log_z_lr", default=1e-3, type=float)
    parser.add_argument("--policy_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1, type=int)
    parser.add_argument("--control_variate", default="global", type=str)
    parser.add_argument("--w_hbond", default=0.0, type=float) 
    args = parser.parse_args()
    args.training = True
    args.save_dir = f"results/{args.date}"

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
    args.h_bond_r0 = 0.3  # Example value, adjust as needed
    args.h_bond_n_min = 6  # Example value, adjust as needed
    args.w_cation = 0.1  # Example value, adjust as needed


    for name in ["policies", "positions", "velocities","forces", "biases"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")
    if args.wandb:
        wandb.init(project="tps-dps", config=args)
    torch.manual_seed(args.seed)
    mds = MDs(args)
    logger = Logger(args, mds)
    agent = DiffusionPathSampler(args, mds)
    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )
    for rollout in range(args.num_rollouts):
        agent.sample(args, mds, temperatures[rollout])
        loss = agent.train(args, mds)
        print(f'{rollout} Loss : {loss}')
        torch.save(agent.policy.state_dict(), f"{args.save_dir}/policies/policy.pt")
        #logger(loss, rollout, agent.policy)


if __name__ == "__main__":
    main()
