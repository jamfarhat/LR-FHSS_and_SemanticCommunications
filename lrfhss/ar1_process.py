"""
Pre-generated AR(1) process for use in traffic generators.

Generates the full discrete-time AR(1) signal at a fixed fine time step (dt_fine)
before the simulation starts. During the simulation, traffic generators query
the process state at any epoch via :meth:`query`.

This allows all protocols (DR8, DR9, Semantic) to be evaluated on exactly the
same underlying physical process, enabling fair comparison and rich visualisation
(continuous x_k(t) curve, step-wise x̂_k(t), real inter-epoch distortion).

Usage
-----
    from lrfhss.ar1_process import AR1Process

    proc = AR1Process(alpha=0.95, sigma_w=1.0, sim_time=3600, dt_fine=1.0, seed=42)
    x_at_t = proc.query(1234.5)          # state at t = 1234.5 s (linear interpolation)
    t_arr, x_arr = proc.full_trace()     # complete time/value arrays
    proc.reset()                          # reuse with same pre-generated signal
"""

from __future__ import annotations

import numpy as np


class AR1Process:
    """
    Pre-generated AR(1) process.

    Parameters
    ----------
    alpha : float
        AR(1) autocorrelation coefficient (0 < alpha < 1).
    sigma_w : float
        Standard deviation of the innovation noise.
    sim_time : float
        Total simulation duration in seconds.
    dt_fine : float
        Time resolution for the pre-generated signal (seconds). Default 1.0 s.
    seed : int or None
        Random seed for reproducibility.
    x0 : float or None
        Initial state. If None, drawn from the stationary distribution N(0, σ_w²).
    """

    def __init__(
        self,
        alpha: float = 0.95,
        sigma_w: float = 1.0,
        sim_time: float = 3600.0,
        dt_fine: float = 1.0,
        lambda_ref: float = 300.0,
        seed: int | None = None,
        x0: float | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.sigma_w = float(sigma_w)
        self.sim_time = float(sim_time)
        self.dt_fine = float(dt_fine)
        self.lambda_ref = float(lambda_ref)   # epoch length the alpha was defined for
        self._seed = seed

        self._t: np.ndarray | None = None
        self._x: np.ndarray | None = None

        self._generate(x0=x0)

    # ── Generation ────────────────────────────────────────────────────────

    def _generate(self, x0: float | None = None) -> None:
        """Pre-generate the full signal."""
        rng = np.random.default_rng(self._seed)

        n_steps = int(np.ceil(self.sim_time / self.dt_fine)) + 1
        t = np.arange(n_steps) * self.dt_fine

        x = np.empty(n_steps)
        x[0] = float(x0) if x0 is not None else rng.normal(0.0, self.sigma_w)

        # alpha is defined per epoch of length lambda_ref seconds.
        # For a fine step of dt_fine seconds the effective per-step coefficient is:
        #   alpha_step = alpha ^ (dt_fine / lambda_ref)
        # Similarly the innovation variance scales linearly with dt_fine:
        #   sigma_step = sigma_w * sqrt(dt_fine / lambda_ref)
        dt_ratio   = self.dt_fine / self.lambda_ref
        alpha_step = self.alpha ** dt_ratio
        sigma_step = self.sigma_w * np.sqrt(dt_ratio)

        noise = rng.normal(0.0, sigma_step, size=n_steps - 1)
        for i in range(1, n_steps):
            x[i] = alpha_step * x[i - 1] + noise[i - 1]

        self._t = t
        self._x = x

    # ── Public API ────────────────────────────────────────────────────────

    def query(self, t: float) -> float:
        """
        Return the process value at time *t* (seconds) via linear interpolation.

        Parameters
        ----------
        t : float
            Query time. Clamped to [0, sim_time].
        """
        t_clamped = float(np.clip(t, 0.0, self.sim_time))
        return float(np.interp(t_clamped, self._t, self._x))

    def full_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (t_array, x_array) for the complete pre-generated signal."""
        return self._t.copy(), self._x.copy()

    def reset(self) -> None:
        """Regenerate signal with the same seed (idempotent replay)."""
        self._generate()

    @property
    def t_array(self) -> np.ndarray:
        return self._t

    @property
    def x_array(self) -> np.ndarray:
        return self._x
