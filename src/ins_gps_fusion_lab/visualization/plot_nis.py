"""Plot NIS (Normalized Innovation Squared) time series.

NIS ~ chi-squared(dof) under consistency. Plots NIS with upper/lower bounds
and marks outliers.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_nis(
    nis_history: np.ndarray,
    dof: int,
    alpha: float = 0.05,
    t: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    mark_outliers: bool = True,
) -> plt.Axes:
    """Plot NIS time series with chi-squared consistency bounds.

    Parameters
    ----------
    nis_history : np.ndarray
        NIS values (1D or 2D with second dim = time). If 2D, use (n_runs, n_steps).
    dof : int
        Degrees of freedom (dimension of measurement).
    alpha : float
        Significance level for bounds. Lower bound at alpha/2, upper at 1-alpha/2.
    t : np.ndarray or None
        Time axis. If None, use indices.
    ax : matplotlib Axes or None
        Axes to plot on. If None, create new figure.
    mark_outliers : bool
        If True, mark points outside bounds.

    Returns
    -------
    ax : matplotlib Axes
    """
    nis = np.asarray(nis_history)
    if nis.ndim == 2:
        nis_flat = nis.flatten()
    else:
        nis_flat = nis.ravel()
    n = len(nis_flat)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if t is None:
        t = np.arange(n)

    # Chi-squared bounds
    lower = stats.chi2.ppf(alpha / 2, dof)
    upper = stats.chi2.ppf(1 - alpha / 2, dof)

    ax.plot(t, nis_flat, "b-", alpha=0.7, label="NIS")
    ax.axhline(y=upper, color="r", linestyle="--", alpha=0.8, label=f"Upper bound (alpha={alpha})")
    ax.axhline(y=lower, color="r", linestyle="--", alpha=0.8)
    ax.axhline(y=dof, color="g", linestyle=":", alpha=0.6, label=f"Mean (dof={dof})")

    if mark_outliers:
        outliers = (nis_flat < lower) | (nis_flat > upper)
        if np.any(outliers):
            ax.scatter(t[outliers], nis_flat[outliers], c="red", s=20, zorder=5, label="Outlier")

    ax.set_xlabel("Time (step)")
    ax.set_ylabel("NIS")
    ax.set_title("Normalized Innovation Squared (Consistency Check)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
