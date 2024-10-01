import os
import torch
import argparse

from flow import FlowNetAgent
from utils.logging import Log
from dynamics import Synthetic

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default="cuda", type=str)

# Logger Config
parser.add_argument("--save_dir", default="results/synthetic", type=str)

# Policy Config
parser.add_argument("--bias", default="pot", type=str)

# Sampling Config
parser.add_argument("--sigma", default=3, type=float)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=0.01, type=float)
parser.add_argument("--num_samples", default=512, type=int)
parser.add_argument("--temperature", default=1200, type=float)

# Training Config
parser.add_argument("--start_temperature", default=2400, type=float)
parser.add_argument("--end_temperature", default=1200, type=float)
parser.add_argument("--num_rollouts", default=20, type=int)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--log_z_lr", default=1e-3, type=float)
parser.add_argument(
    "--policy_lr",
    default=1e-4,
    type=float,
)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument(
    "--buffer_size",
    default=10000,
    type=int,
)
parser.add_argument(
    "--trains_per_rollout",
    default=1000,
    type=int,
)

args = parser.parse_args()

if __name__ == "__main__":
    args.training = True
    for name in ["policies", "positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    torch.manual_seed(args.seed)

    mds = Synthetic(args)
    log = Log(args, mds)
    agent = FlowNetAgent(args, mds)

    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )

    stds = torch.sqrt(2 * mds.kB * args.timestep * temperatures)

    log.info("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")

        agent.sample(args, mds, stds[rollout])
        log.sample(rollout, agent.policy)
        loss = agent.train(args, mds)
        log.logger.info(f"loss: {loss}")
    log.info("End training")
