import argparse

from utils.plot import Plot
from dynamics import Synthetic
from utils.metrics import Metric

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--save_dir", default="paths/synthetic", type=str)

# Sampling Config
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=0.01, type=float)
parser.add_argument("--num_samples", default=1024, type=int)
parser.add_argument("--temperature", default=1200, type=float)

args = parser.parse_args()


class Eval:
    def __init__(self, args):
        mds = Synthetic(args)

        self.plot = Plot(args, mds)
        self.metric = Metric(args, mds)

    def __call__(self):
        log = {}
        log.update(self.metric())

        print(
            f"RMSD: {log['rmsd']:.2f} ± {log['rmsd_std']:.2f} "
            f"THP: {log['thp']:.2f} "
            f"ETP: {log['etp']:.2f} ± {log['etp_std']:.2f}"
        )

        self.plot()


if __name__ == "__main__":
    Eval(args)()
