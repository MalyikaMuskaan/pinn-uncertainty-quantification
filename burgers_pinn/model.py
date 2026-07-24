"""
model.py
--------
Defines the Physics-Informed Neural Network (PINN) architecture for solving
the 1D Burgers' equation.

The network maps (x, t) -> u, i.e. it takes spatial coordinate x and time t
as inputs and outputs the predicted velocity field u(x, t).

Architecture choice rationale:
  - tanh activations are smooth and infinitely differentiable, which is
    critical because the PDE loss requires computing ∂u/∂x, ∂u/∂t, and
    ∂²u/∂x² via automatic differentiation.
  - Xavier (Glorot) initialisation keeps gradients well-scaled at the start
    of training for tanh networks.
"""

import torch
import torch.nn as nn


class BurgersPINN(nn.Module):
    """
    Fully-connected neural network approximating u(x, t) for the Burgers' equation.

    Architecture
    ------------
    Input  : (x, t)  — shape (N, 2)
    Hidden : n_hidden layers, each with n_neurons neurons + tanh activation
    Output : u        — shape (N, 1)
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 50):
        """
        Parameters
        ----------
        n_hidden  : number of hidden layers (default 4, minimum required)
        n_neurons : neurons per hidden layer  (default 50)
        """
        super().__init__()

        # Build the layer list dynamically so n_hidden / n_neurons are flexible.
        layers = []

        # --- Input layer: 2 inputs (x, t) -> first hidden layer ---
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())

        # --- Hidden layers ---
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # --- Output layer: hidden -> 1 output (u) ---
        # No activation — the output is an unbounded scalar field.
        layers.append(nn.Linear(n_neurons, 1))

        self.network = nn.Sequential(*layers)

        # Apply Xavier initialisation to all Linear layers.
        self._initialise_weights()

    def _initialise_weights(self):
        """Xavier (Glorot) uniform initialisation for every Linear layer."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : tensor of shape (N, 1) — spatial coordinates
        t : tensor of shape (N, 1) — time coordinates

        Returns
        -------
        u : tensor of shape (N, 1) — predicted solution
        """
        # Concatenate inputs along the feature dimension -> (N, 2)
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)
