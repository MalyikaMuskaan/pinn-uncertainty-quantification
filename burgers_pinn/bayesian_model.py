"""
bayesian_model.py
-----------------
Bayesian Neural Network (BNN) for the Burgers' PINN using mean-field
Variational Inference (VI), also known as "Bayes by Backprop"
(Blundell et al., 2015).

Architecture
------------
Identical to BurgersPINN in model.py (same depth / width / tanh activations)
so the two approaches are fairly comparable:
  Input  : (x, t)   → 2 neurons
  Hidden : n_hidden layers of n_neurons neurons each, tanh activation
  Output : u(x, t)  → 1 neuron

Key Bayesian concept — mean-field VI
-------------------------------------
Instead of a single weight w for each connection, we learn two parameters:
  mu    : variational mean  (unconstrained scalar)
  rho   : log-softplus parameter so  sigma = softplus(rho) > 0 always

At every forward pass we sample:
  w ~ N(mu, sigma^2)     via the reparameterisation trick:
  w = mu + sigma * eps,  eps ~ N(0, 1)

This keeps the sampling differentiable, so we can back-propagate through the
stochastic weights.

KL divergence
-------------
Each layer contributes a KL term measuring how far the learned posterior
q(w | mu, sigma) is from the prior p(w) = N(0, 1):
  KL[q || p] = sum over weights of:
    log(sigma_prior / sigma) + (sigma^2 + mu^2) / (2 * sigma_prior^2) - 0.5

The total KL is accumulated across all BayesLinear layers and returned as a
scalar alongside the network output so the training loop can weight it.

Rho initialisation
------------------
We initialise rho so that the initial sigma ~ 0.1 (a moderately tight
posterior), which prevents the KL from completely overwhelming the data-fit
terms at the start of training.  softplus(rho) = 0.1 → rho ≈ -2.25.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Prior standard deviation for all weights (N(0, PRIOR_SIGMA^2))
PRIOR_SIGMA = 1.0


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softplus: log(1 + exp(x))."""
    return F.softplus(x)


# ------------------------------------------------------------------ #
#  Bayesian Linear Layer                                               #
# ------------------------------------------------------------------ #

class BayesLinear(nn.Module):
    """
    A linear layer where weights and biases are Gaussian random variables.

    Parameters (learnable)
    ----------------------
    weight_mu  : (out, in)  — posterior mean of weights
    weight_rho : (out, in)  — posterior log-softplus of weight std
    bias_mu    : (out,)     — posterior mean of biases
    bias_rho   : (out,)     — posterior log-softplus of bias std

    At each forward call a fresh weight sample is drawn via the
    reparameterisation trick, and the KL[q||p] contribution is computed.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Posterior mean — initialised with Xavier uniform scale
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features))

        # Posterior rho — initialised so initial sigma ≈ 0.1
        # softplus(rho) = 0.1  =>  rho = log(exp(0.1) - 1) ≈ -2.2504
        init_rho = math.log(math.expm1(0.1))   # ≈ -2.2504
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), init_rho)
        )
        self.bias_rho   = nn.Parameter(
            torch.full((out_features,), init_rho)
        )

        # Xavier initialisation for the means
        nn.init.xavier_uniform_(self.weight_mu)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample weights, compute linear output, and accumulate KL.

        Parameters
        ----------
        x : (N, in_features)

        Returns
        -------
        out : (N, out_features)  — sampled linear output
        kl  : scalar tensor       — KL[q(w,b) || p(w,b)] for this layer
        """
        # ---- Reparameterisation trick for weights ----
        weight_sigma = _softplus(self.weight_rho)            # always > 0
        weight_eps   = torch.randn_like(self.weight_mu)
        weight       = self.weight_mu + weight_sigma * weight_eps

        # ---- Reparameterisation trick for biases ----
        bias_sigma = _softplus(self.bias_rho)
        bias_eps   = torch.randn_like(self.bias_mu)
        bias       = self.bias_mu + bias_sigma * bias_eps

        # ---- Forward linear pass ----
        out = F.linear(x, weight, bias)

        # ---- KL divergence contribution (closed form for Gaussians) ----
        kl = _kl_gaussian(self.weight_mu, weight_sigma) \
           + _kl_gaussian(self.bias_mu,   bias_sigma)

        return out, kl

    def kl_divergence(self) -> torch.Tensor:
        """Return KL without a forward pass (for diagnostic use)."""
        weight_sigma = _softplus(self.weight_rho)
        bias_sigma   = _softplus(self.bias_rho)
        return _kl_gaussian(self.weight_mu, weight_sigma) \
             + _kl_gaussian(self.bias_mu,   bias_sigma)


def _kl_gaussian(mu: torch.Tensor, sigma: torch.Tensor,
                 prior_sigma: float = PRIOR_SIGMA) -> torch.Tensor:
    """
    Closed-form KL[N(mu, sigma^2) || N(0, prior_sigma^2)] summed over all
    elements of mu / sigma.

    KL = sum [ log(prior_sigma/sigma) + (sigma^2 + mu^2)/(2*prior_sigma^2) - 0.5 ]
    """
    ps2 = prior_sigma ** 2
    kl  = (math.log(prior_sigma) - torch.log(sigma)
           + (sigma ** 2 + mu ** 2) / (2.0 * ps2)
           - 0.5)
    return kl.sum()


# ------------------------------------------------------------------ #
#  Bayesian PINN                                                       #
# ------------------------------------------------------------------ #

class BayesianBurgersPINN(nn.Module):
    """
    Bayesian PINN for Burgers' equation using mean-field VI.

    Same architecture as BurgersPINN (model.py) but every Linear layer is
    replaced by a BayesLinear layer.  The forward pass draws a fresh weight
    sample and returns both the prediction and the accumulated KL term.

    Parameters
    ----------
    n_hidden  : number of hidden layers (default 4)
    n_neurons : neurons per hidden layer  (default 50)
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 50):
        super().__init__()

        self.n_hidden  = n_hidden
        self.n_neurons = n_neurons

        # Build list of BayesLinear layers (stored in a ModuleList so
        # their parameters are registered with the optimiser automatically)
        layers = [BayesLinear(2, n_neurons)]
        for _ in range(n_hidden - 1):
            layers.append(BayesLinear(n_neurons, n_neurons))
        layers.append(BayesLinear(n_neurons, 1))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic forward pass — draws one weight sample per call.

        Parameters
        ----------
        x : (N, 1) spatial coordinates
        t : (N, 1) time coordinates

        Returns
        -------
        u   : (N, 1) predicted solution for this weight sample
        kl  : scalar  total KL across all layers
        """
        h   = torch.cat([x, t], dim=1)   # (N, 2)
        kl  = torch.zeros(1, device=x.device)

        # Hidden layers (all but the last) — apply tanh after each
        for layer in self.layers[:-1]:
            h, layer_kl = layer(h)
            h   = torch.tanh(h)
            kl  = kl + layer_kl

        # Output layer — no activation
        u, layer_kl = self.layers[-1](h)
        kl = kl + layer_kl

        return u, kl

    def predict_mean(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Deterministic forward using posterior means only (no sampling).
        Useful for a quick point estimate without randomness.
        """
        h = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            # Use mean weights directly
            w  = layer.weight_mu
            b  = layer.bias_mu
            h  = torch.tanh(F.linear(h, w, b))
        w = self.layers[-1].weight_mu
        b = self.layers[-1].bias_mu
        return F.linear(h, w, b)

    def total_kl(self) -> torch.Tensor:
        """Sum KL over all layers without a forward pass."""
        kl = torch.zeros(1, device=self.layers[0].weight_mu.device)
        for layer in self.layers:
            kl = kl + layer.kl_divergence()
        return kl
