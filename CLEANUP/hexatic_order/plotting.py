# hexatic_order/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import (
    Union,
    Sequence,
    Optional,
    Tuple,
)
from .order import g6

def compute_eta(
    r: np.ndarray,
    g6: np.ndarray,
    rmin: float,
    rmax: float,
    A0: float
) -> float:
    """Least-squares log-log fit of g₆(r) ~ A0 * r^{–η} over [rmin,rmax]."""
    mask = (r >= rmin) & (r <= rmax) & (g6 > 0)
    rf, gf = r[mask], g6[mask]
    X = np.log(rf)
    Y = np.log(gf) - np.log(A0)
    return -np.sum(X * Y) / np.sum(X * X)

def compute_xi(
    r: np.ndarray,
    g6: np.ndarray,
    rmin: float,
    rmax: float,
    A0: float,
    offset: float = 1.0
) -> float:
    """Least-squares fit ln[g₆(r)/A0] ~ -(r-offset)/ξ over [rmin,rmax]."""
    mask = (r >= rmin) & (r <= rmax) & (g6 > 0)
    rf, gf = r[mask], g6[mask]
    X = rf - offset
    Y = np.log(gf) - np.log(A0)
    inv_xi = -np.sum(X * Y) / np.sum(X * X)
    return 1.0 / inv_xi

def _prepare_curve(
    pos: np.ndarray,
    dr: float,
    rmin: float,
    rmax: float,
    sigma: float,
    threshold_eta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Compute and smooth g6(r), compute A0, η, decide fit, and
    return (r, g6_sm, rk, gk, fit_type, fit_param).
    """
    # raw
    r, g6_raw, _ = g6(pos, dr=dr)
    # smooth
    g6_sm = gaussian_filter1d(np.nan_to_num(g6_raw, nan=0.0), sigma=sigma)
    # amplitude at r≈1.0
    idx1 = np.argmin(np.abs(r - 1.0))
    A0 = g6_sm[idx1]
    # compute η
    η = compute_eta(r, g6_sm, rmin, rmax, A0)

    # prepare fit curve grid
    rk = np.linspace(rmin, rmax, 200)
    if η <= threshold_eta:
        fit_type = "algebraic"
        gk = A0 * rk**(-η)
        fit_param = η
    else:
        fit_type = "exponential"
        ξ = compute_xi(r, g6_sm, rmin, rmax, A0, offset=1.0)
        gk = A0 * np.exp(- (rk - 1.0) / ξ)
        fit_param = ξ

    return r, g6_sm, rk, gk, fit_type, fit_param

def plot_g6(
    positions: Union[np.ndarray, Sequence[np.ndarray]],
    dr: float = 0.05,
    rmin: float = 1.0,
    rmax: float = 50.0,
    sigma: float = 2.0,
    threshold_eta: float = 0.25,
    colors: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot one or more g₆(r) curves (with smoothing & fits) on the same log-log plot.

    Parameters
    ----------
    positions
      Either a single (N,2) array, or a list of arrays [(N1,2), (N2,2), ...].
    dr
      bin width for computing g₆(r).
    rmin, rmax
      fitting window for η or ξ.
    sigma
      Gaussian smoothing sigma.
    threshold_eta
      if computed η > threshold_eta, do exponential fit; else algebraic fit.
    colors
      list of colors (must match number of trajectories). Defaults to matplotlib cycle.
    labels
      list of labels for the legend.
    ax
      existing matplotlib Axes to draw into.

    Returns
    -------
    ax
      The Axes object containing the plot.
    """
    # Normalize to list
    if isinstance(positions, np.ndarray):
        pos_list = [positions]
    else:
        pos_list = positions  # assume sequence of arrays

    n = len(pos_list)
    # default colors/labels
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if labels is None:
        labels = [None] * n

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6), dpi=600)

    for i, pos in enumerate(pos_list):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else None

        # Prepare the data + fits
        r, g6_sm, rk, gk, fit_type, fit_param = _prepare_curve(
            pos, dr, rmin, rmax, sigma, threshold_eta
        )

        # Plot raw g6
        ax.loglog(
            r, g6_sm,
            c=color, alpha=0.6, lw=1.5,
        )
        # Plot the fit
        if fit_type == 'algebraic' and fit_param < 0.02:
            fit_label = rf"crystalline: $\eta_6$={np.abs(fit_param):.2f}"

        elif fit_type == 'algebraic' and fit_param > 0.02:
            fit_label = rf"hexatic: $\eta_6$={np.abs(fit_param):.2f}"
        
        else:
            fit_label = rf"isotropic liquid: $\xi_6$={np.abs(fit_param):.2f}"
        ax.loglog(
            rk, gk,
            '--', c=color, alpha=1.0, lw=1.5,
            label=f"{fit_label}"
        )

    ax.set_xlabel(r"$r/a$",fontsize=20)
    ax.set_ylabel(r"$g_6(r)$", fontsize=20)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(1e-2, 1.05)
    ax.grid(True, which='both', ls='--')
    ax.legend()
    return ax