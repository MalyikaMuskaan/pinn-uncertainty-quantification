"""
model.py
--------
Inverse-problem PINN for the viscous Burgers' equation.

The key difference from burgers_pinn/model.py: the kinematic viscosity ν
is a learnable nn.Parameter instead of a fixed constant.  The optimiser
updates it alongside the network weights every epoch.

To prevent the optimiser from pulling ν to negative values (which would
make the PDE ill-posed), we store an unconstrained raw variable and expose
the physical ν via a softplus transformation:

    ν = log(1 + exp(raw_nu))    (always positive, smooth gradient)

The initial raw_nu is chosen so that ν starts at `nu_init`, which should
be intentionally wrong (e.g., 2–5× the true value) to test recovery.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseBurgersPINN(nn.Module):
    """
    Fully-connected PINN approximating u(x,t) for the inverse Burgers problem.

    Architecture
    ------------
    Input  : (x, t) — shape (N, 2)
    Hidden : n_hidden layers, n_neurons neurons each, tanh activation
    Output : u       — shape (N, 1)

    Parameters
    ----------
    n_hidden : number of hidden layers (default 4)
    n_neurons: neurons per hidden layer (default 50)
    nu_init  : initial guess for ν (intentionally wrong)
    """

    def __init__(
        self,
        n_hidden:  int   = 4,
        n_neurons: int   = 50,
        nu_init:   float = 0.1,   # wrong initial guess (true ≈ 0.003183)
    ):
        super().__init__()

        # ---- Solution network u(x,t) ----
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

        # ---- Learnable viscosity ν (stored as unconstrained raw_nu) ----
        # softplus(raw_nu) = nu  →  raw_nu = log(exp(nu_init) - 1)
        # For numerical safety, clamp nu_init to be > 1e-6
        nu_init = max(nu_init, 1e-6)
        raw_init = math.log(math.expm1(nu_init))   # inverse softplus
        self.raw_nu = nn.Parameter(torch.tensor([raw_init], dtype=torch.float32))

    # -------------------------------------------------------------- #
    #  Property: physical ν (always positive)                         #
    # -------------------------------------------------------------- #
    @property
    def nu(self) -> torch.Tensor:
        """Positive viscosity via softplus: ν = log(1 + exp(raw_nu))."""
        return F.softplus(self.raw_nu)

    # -------------------------------------------------------------- #
    #  Convenience scalar for logging / storage                        #
    # -------------------------------------------------------------- #
    def nu_value(self) -> float:
        """Return the current ν estimate as a plain Python float."""
        return float(self.nu.item())

    # -------------------------------------------------------------- #
    #  Weight initialisation                                           #
    # -------------------------------------------------------------- #
    def _init_weights(self):
        """Xavier (Glorot) uniform for all Linear layers."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------- #
    #  Forward pass                                                    #
    # -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1) spatial coordinates
        t : (N, 1) time coordinates

        Returns
        -------
        u : (N, 1) predicted solution
        """
        return self.network(torch.cat([x, t], dim=1))
