"""Plot covariance evolution over time.

Visualizes P trace, diagonal elements, or 2D position uncertainty ellipse.
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_covariance(
    P_history: Union[list[np.ndarray], np.ndarray],
    indices: Optional[list[int]] = None,
    t: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    plot_trace: bool = True,
) -> plt.Axes:
    """Plot covariance matrix evolution over time.

    Parameters
    ----------
    P_history : list of np.ndarray or np.ndarray
        Sequence of covariance matrices (n, n) per step.
        If 3D array, shape (n_steps, n, n).
    indices : list of int or None
        Indices of diagonal elements to plot (e.g., [0,1,2] for position variances).
        If None and plot_trace=True, plot trace only.
    t : np.ndarray or None
        Time axis. If None, use step indices.
    ax : matplotlib Axes or None
        Axes to plot on. If None, create new figure.
    plot_trace : bool
        If True, plot trace(P) as main curve. If indices is also set, plot both.

    Returns
    -------
    ax : matplotlib Axes
    """
    if isinstance(P_history, list):
        P_arr = np.array(P_history)
    else:
        P_arr = np.asarray(P_history)

    if P_arr.ndim == 2:
        P_arr = P_arr[np.newaxis, ...]
    n_steps = P_arr.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    if t is None:
        t = np.arange(n_steps)

    if plot_trace:
        trace = np.array([np.trace(P) for P in P_arr])
        ax.plot(t, trace, "b-", label="trace(P)", linewidth=2)

    if indices is not None:
        for _i, idx in enumerate(indices):
            diag = np.array([P_arr[k, idx, idx] for k in range(n_steps)])
            ax.plot(t, diag, "--", alpha=0.8, label=f"P[{idx},{idx}]")

    ax.set_xlabel("Time (step)")
    ax.set_ylabel("Covariance")
    ax.set_title("Covariance Evolution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_position_ellipse(
    P: np.ndarray,
    x: np.ndarray,
    n_sigma: float = 2.0,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot 2D position uncertainty ellipse from P[0:2, 0:2].

    Parameters
    ----------
    P : np.ndarray (n, n)
        Covariance matrix.
    x : np.ndarray (n,)
        State (position in x[0:2]).
    n_sigma : float
        Number of standard deviations for ellipse.
    ax : matplotlib Axes or None
    **kwargs : passed to plot/patches

    Returns
    -------
    ax : matplotlib Axes
    """
    from matplotlib.patches import Ellipse

    P_pos = P[0:2, 0:2]
    eigvals, eigvecs = np.linalg.eigh(P_pos)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_sigma * np.sqrt(eigvals[0])
    height = 2 * n_sigma * np.sqrt(eigvals[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ell = Ellipse(
        xy=(x[0], x[1]),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs,
    )
    ax.add_patch(ell)
    ax.plot(x[0], x[1], "k.")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax
