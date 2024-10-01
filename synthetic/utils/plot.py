import torch
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, args, mds):
        self.device = args.device
        self.save_dir = args.save_dir
        self.num_samples = args.num_samples
        self.start_position = mds.start_position
        self.target_position = mds.target_position
        self.energy_function = mds.energy_function

    def __call__(self):
        positions = []
        for i in range(64):
            position = np.load(f"{self.save_dir}/positions/{i}.npy")
            position = torch.from_numpy(position).to(self.device)
            positions.append(position)

        self.paths(positions)

    def paths(self, positions):
        fig, ax = plt.subplots(figsize=(7, 7))

        zorder = 100
        circle_size = 1200
        saddle_size = 2400

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        x = np.linspace(-1.5, 1.5, 400)
        y = np.linspace(-1.5, 1.5, 400)
        X, Y = np.meshgrid(x, y)

        Z = np.load(f"data/synthetic/landscape.npy")

        ax.contourf(X, Y, Z, levels=zorder, zorder=0, vmax=3)

        cm = plt.get_cmap("gist_rainbow")

        ax.set_prop_cycle(
            color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
        )

        for position in positions[:64]:
            xs = position[:, 0].detach().cpu().numpy()
            ys = position[:, 1].detach().cpu().numpy()
            ax.plot(
                xs,
                ys,
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
                zorder=zorder - 1,
            )

        # Plot start and end positions
        ax.scatter(
            [self.start_position[0].item()],
            [self.start_position[1].item()],
            edgecolors="black",
            c="w",
            zorder=zorder,
            s=circle_size,
        )
        ax.scatter(
            [self.target_position[0].item()],
            [self.target_position[1].item()],
            edgecolors="black",
            c="w",
            zorder=zorder,
            s=circle_size,
        )

        saddle_points = [(0, 1), (0, -1)]
        for saddle in saddle_points:
            ax.scatter(
                saddle[0],
                saddle[1],
                edgecolors="black",
                c="w",
                zorder=zorder,
                s=saddle_size,
                marker="*",
            )

        # Plot basic configs
        ax.set_xlabel("x", fontsize=24, fontweight="medium")
        ax.set_ylabel("y", fontsize=24, fontweight="medium")
        plt.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            bottom=False,
        )
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/paths.png")
        plt.show()
        plt.close()
        return fig
