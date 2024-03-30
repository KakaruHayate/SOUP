import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class PitchNet(nn.Module):
    def __init__(self, in_dim=1025, out_dim=256, kernel=5, n_layers=3, strides=None):
        super().__init__()

        self.in_linear = nn.Sequential(
            nn.Linear(1, 16),
            Mish(),
            nn.Linear(16, 64),
            Mish(),
            nn.Linear(64, 256),
            Mish(),
            nn.Linear(256, 1025),
        )

        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for l in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=kernel,
                        padding=padding,
                        stride=self.strides[l],
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                )
            )
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim // 4),
            Mish(),
            nn.Linear(out_dim // 4, out_dim // 16),
            Mish(),
            nn.Linear(out_dim // 16, out_dim // 64),
            Mish(),
            nn.Linear(out_dim // 64, 1),
        )
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, sp_h, midi):
        """
        sp_h:[B,M,1025]
        midi:[B,M,1]
        output:[B,M,]
        """
        midi = self.in_linear(midi)

        x = torch.cat([midi, sp_h], dim=1)

        x = sp_h.transpose(1, 2)
        

        for _, l in enumerate(self.layers):
            x = l(x)

        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.reshape(x.shape[0], x.shape[1])
        #print(x.shape)

        return x
