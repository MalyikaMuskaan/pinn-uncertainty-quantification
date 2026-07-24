"""
model.py  —  ocean_pinn
-----------------------
Defines the PINN architecture for solving the 1D advection-diffusion equation:

    dc/dt + v*(dc/dx) = D*(d²c/dx²)

The network maps (x, t) -> c(x, t), i.e. takes spatial coordinate x (in km)
and time t as inputs and outputs the predicted concentration field c.

Architecture is identical to burgers_pinn/model.py (same depth, width, tanh
activations, Xavier initialisation) so comparisons between problems are fair.
"""

import torch
import torch.nn as nn


class OceanPINN(nn.Module):
    """
    Fully-connected PINN for the 1D advection-diffusion equation.

    Input  : (x, t)  — shape (N, 2)
    Hidden : n_hidden layers of n_neurons neurons each + tanh activation
    Output : c(x, t) — shape (N, 1)

    tanh is chosen because the PDE residual requires dc/dx, dc/dt, d²c/dx²
    via autograd, so the activation must be smooth and infinitely differentiable.
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 50):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())

        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Output — no activation (concentration is a non-negative scalar,
        # but we leave it unconstrained and let the physics enforce positivity)
        layers.append(nn.Linear(n_neurons, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for every Linear layer; zero biases."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1) spatial coordinates  [km, in [0, 10]]
        t : (N, 1) time coordinates      [in [0,  5]]

        Returns
        -------
        c : (N, 1) predicted concentration
        """
        return self.network(torch.cat([x, t], dim=1))
