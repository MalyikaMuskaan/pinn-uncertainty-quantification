"""
model.py
--------
Physics-Informed Neural Network for the 2-D steady-state Darcy flow equation:

    -div( k(x,y) * grad(u(x,y)) ) = f(x,y)   on [0,1] x [0,1]
    u = 0                                       on all four boundaries

The network maps (x, y) → u̲(x, y) — a scalar pressure/head field.

Architecture choice
-------------------
- 2D input replaces the (x,t) input of the 1-D PINNs: the network takes
  (x, y) and outputs one scalar u.
- tanh activations: smooth, infinitely differentiable — required because the
  PDE loss involves ∂u/∂x, ∂u/∂y, and their second derivatives via autograd.
- 5 hidden layers × 64 neurons: deeper than the 1-D Burgers PINN to handle
  the 2-D spatial variation of k(x,y) and the coupling between x and y in
  the mixed-derivative PDE operator.
- Xavier (Glorot) uniform initialisation: keeps gradient magnitudes at init
  well-scaled for tanh networks.
"""

import torch
import torch.nn as nn


class DarcyPINN(nn.Module):
    """
    Fully-connected PINN approximating u(x, y) for 2-D Darcy flow.

    Architecture
    ------------
    Input  : (x, y) — shape (N, 2)
    Hidden : n_hidden layers, each n_neurons neurons + tanh activation
    Output : u       — shape (N, 1)

    Parameters
    ----------
    n_hidden  : number of hidden layers (default 5)
    n_neurons : neurons per hidden layer (default 64)
    """

    def __init__(self, n_hidden: int = 5, n_neurons: int = 64):
        super().__init__()

        layers: list[nn.Module] = []

        # Input layer: 2 inputs (x, y) → first hidden layer
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())

        # Additional hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Output layer: no activation — u is an unbounded scalar field
        layers.append(nn.Linear(n_neurons, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Xavier (Glorot) uniform initialisation for every Linear layer."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (N, 1) tensor — x-coordinates in [0, 1]
        y : (N, 1) tensor — y-coordinates in [0, 1]

        Returns
        -------
        u : (N, 1) tensor — predicted solution field
        """
        return self.network(torch.cat([x, y], dim=1))
