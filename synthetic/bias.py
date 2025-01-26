import torch
from torch import nn
from torch.nn.functional import softplus


class BiasForce(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bias = args.bias

        if self.bias in ["force"]:
            self.output_dim = 2
        elif self.bias in ["pot", "scale"]:
            self.output_dim = 1

        self.input_dim = 2 + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, self.output_dim, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))

        self.to(args.device)

    def forward(self, pos, target):
        if self.bias == "pot":
            pos.requires_grad = True

        dist = torch.norm(pos - target, dim=-1, keepdim=True)
        pos_ = torch.cat([pos, dist], dim=-1)

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if self.bias == "force":
            force = out.view(*pos.shape)
        elif self.bias == "pot":
            force = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        elif self.bias == "scale":
            scale = softplus(out).view(*pos.shape[:-1], self.output_dim)
            force = scale * (target - pos)
        return force
