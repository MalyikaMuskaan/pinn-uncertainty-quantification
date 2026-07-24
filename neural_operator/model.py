"""
model.py
--------
1-D Fourier Neural Operator (FNO) for the viscous Burgers' equation.

Task
----
Learn the operator  G : u₀(x) → u(x, t)  that maps an initial condition
to the full spatio-temporal solution field.

    Input  shape : (batch, N_x)          — initial condition on the spatial grid
    Output shape : (batch, N_x, N_t)     — solution on the full (x, t) grid

Architecture
------------
Following Li et al. (2020) "Fourier Neural Operator for Parametric PDEs":

    1. Lifting layer   : pointwise linear  N_x → (N_x, width)
    2. Four Fourier layers (SpectralConv1d + pointwise bypass + GELU)
    3. Projection layer: pointwise linear  width → N_t  (at each x location)

Each Fourier layer:
    - Truncates to the lowest `modes` Fourier modes in x
    - Applies a learnable complex weight matrix in spectral space
    - Back-transforms to physical space
    - Adds a bypass linear (W·v, no FFT) for the residual
    - Applies GELU activation

Dimensions / notation
---------------------
  N_x    : number of spatial grid points (default 256)
  N_t    : number of temporal grid points (default 100)
  modes  : number of Fourier modes retained (default 16)
  width  : channel width of the FNO trunk (default 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Spectral convolution layer                                          #
# ------------------------------------------------------------------ #

class SpectralConv1d(nn.Module):
    """
    1-D spectral convolution: multiply in Fourier space, then iFFT.

    Input/output shapes: (batch, width, N_x)
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes   # number of low-frequency modes to keep

        # Complex weights: (width_in, width_out, modes)
        # Stored as real-valued (width, width, modes, 2) for torch compat
        scale = 1.0 / (width * width)
        self.weights_real = nn.Parameter(
            scale * torch.randn(width, width, modes))
        self.weights_imag = nn.Parameter(
            scale * torch.randn(width, width, modes))

    def _complex_weights(self) -> torch.Tensor:
        """Return weights as complex tensor (width, width, modes)."""
        return torch.complex(self.weights_real, self.weights_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, width, N_x)

        Returns
        -------
        out : (batch, width, N_x)
        """
        batch, width, N_x = x.shape

        # FFT along the spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)          # (batch, width, N_x//2+1)

        # Multiply the low-frequency modes by the learnable weights
        out_ft = torch.zeros(batch, self.width, N_x // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        W = self._complex_weights()                # (width, width, modes)
        # einsum: out[b,j,k] = sum_i  W[i,j,k] * x_ft[b,i,k]
        out_ft[:, :, :self.modes] = torch.einsum(
            "bik,ijk->bjk", x_ft[:, :, :self.modes], W)

        # Inverse FFT back to physical space
        return torch.fft.irfft(out_ft, n=N_x, dim=-1)  # (batch, width, N_x)


# ------------------------------------------------------------------ #
#  Single Fourier layer (spectral conv + bypass + activation)          #
# ------------------------------------------------------------------ #

class FourierLayer(nn.Module):
    """Spectral convolution with a pointwise bypass, followed by GELU."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.bypass   = nn.Conv1d(width, width, kernel_size=1)  # W·v, no FFT

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v : (batch, width, N_x)"""
        return F.gelu(self.spectral(v) + self.bypass(v))


# ------------------------------------------------------------------ #
#  Full FNO                                                            #
# ------------------------------------------------------------------ #

class FNO1d(nn.Module):
    """
    1-D FNO mapping initial conditions to full (x, t) solution fields.

    Parameters
    ----------
    n_x    : spatial grid size (must match the dataset)
    n_t    : temporal grid size (must match the dataset)
    modes  : number of Fourier modes to keep (default 16)
    width  : channel width (default 64)
    depth  : number of Fourier layers (default 4)
    """

    def __init__(
        self,
        n_x:   int = 256,
        n_t:   int = 100,
        modes: int = 16,
        width: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.n_x   = n_x
        self.n_t   = n_t
        self.modes = modes
        self.width = width
        self.depth = depth

        # Lifting: (batch, 1, N_x) → (batch, width, N_x)
        # We pass the IC as a single-channel 1-D signal.
        self.lift = nn.Conv1d(1, width, kernel_size=1)

        # Fourier trunk
        self.layers = nn.ModuleList(
            [FourierLayer(width, modes) for _ in range(depth)]
        )

        # Projection: (batch, width, N_x) → (batch, N_t, N_x)
        # Two-step: width → 128 → N_t at each spatial location.
        self.proj1 = nn.Conv1d(width, 128, kernel_size=1)
        self.proj2 = nn.Conv1d(128, n_t, kernel_size=1)

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u0  : (batch, N_x) initial condition

        Returns
        -------
        u   : (batch, N_x, N_t) predicted solution
        """
        # Add channel dim: (batch, 1, N_x)
        v = u0.unsqueeze(1)

        # Lifting
        v = self.lift(v)                      # (batch, width, N_x)

        # Fourier layers
        for layer in self.layers:
            v = layer(v)                      # (batch, width, N_x)

        # Projection to output channels = N_t
        v = F.gelu(self.proj1(v))             # (batch, 128, N_x)
        v = self.proj2(v)                     # (batch, N_t, N_x)

        # Reorder to (batch, N_x, N_t) to match dataset convention
        return v.permute(0, 2, 1)             # (batch, N_x, N_t)
