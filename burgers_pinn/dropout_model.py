"""
dropout_model.py
----------------
BurgersPINN with MC Dropout layers for uncertainty quantification.

Architecture
------------
Identical to model.py (4 hidden layers, 50 neurons, tanh activations) but
with a nn.Dropout layer inserted **after each tanh activation** in the hidden
layers.  The output layer has no dropout.

Why tanh + dropout, not ReLU + dropout?
-----------------------------------------
PINNs require smooth, infinitely-differentiable activations so that the PDE
residual (which needs ∂²u/∂x²) can be computed via autograd.  tanh satisfies
this.  ReLU has zero second derivatives almost everywhere and breaks the
viscous term in Burgers' equation.

Why low dropout rate (p = 0.05)?
----------------------------------
PINNs are sensitive to dropout for two reasons:
  1. The PDE residual enforces global smoothness constraints — heavy dropout
     produces noisy, discontinuous predictions that violate the physics.
  2. With only 50 neurons per layer, aggressive dropout removes too large a
     fraction of the capacity needed to represent the solution.
In practice p=0.05 preserves training accuracy while still producing
meaningful stochastic variation at inference time.

MC Dropout inference
--------------------
At inference time, keep the model in **train() mode** (not eval()).  This
keeps dropout active.  Running N forward passes with different random dropout
masks gives N different predictions; their mean and std are the posterior
predictive mean and uncertainty estimate.

See: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
"""

import torch
import torch.nn as nn


class DropoutBurgersPINN(nn.Module):
    """
    Burgers' PINN with MC Dropout.

    Parameters
    ----------
    n_hidden      : number of hidden layers (default 4)
    n_neurons     : neurons per hidden layer (default 50)
    dropout_rate  : dropout probability applied after each hidden tanh
                    (default 0.05 — kept low for PINN stability)
    """

    def __init__(
        self,
        n_hidden:     int   = 4,
        n_neurons:    int   = 50,
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        layers: list[nn.Module] = []

        # Input → first hidden
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(p=dropout_rate))

        # Remaining hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout_rate))

        # Output — no dropout, no activation
        layers.append(nn.Linear(n_neurons, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation; zero biases."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1) spatial coordinates
        t : (N, 1) time coordinates

        Returns
        -------
        u : (N, 1) predicted solution

        Note: dropout is active when the model is in train() mode.
        Call model.train() before inference for MC Dropout.
        """
        return self.network(torch.cat([x, t], dim=1))

    def enable_mc_dropout(self) -> None:
        """
        Set the model to train() mode so dropout stays active during inference.
        Only Dropout modules are enabled; BatchNorm (absent here) would be
        frozen.  This is the standard MC Dropout inference trick.
        """
        self.train()   # activates all nn.Dropout layers

    def disable_dropout(self) -> None:
        """Set to eval() mode — dropout disabled, deterministic predictions."""
        self.eval()
