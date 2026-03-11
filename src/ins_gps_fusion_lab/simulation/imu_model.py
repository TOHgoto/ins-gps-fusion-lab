"""IMU error model for sensor fusion simulation.

Mathematical model:
-------------------
Accelerometer:  a_meas = a_true + b_a + eta_a    where  b_a  evolves as random walk
Gyroscope:      w_meas = w_true + b_g + eta_g   where  b_g  evolves as random walk

Continuous time:
  b_a_dot = eta_ba   (accel bias random walk)
  b_g_dot = eta_bg   (gyro bias random walk)

Discrete time (Euler):
  b_a[k+1] = b_a[k] + sqrt(dt) * sigma_acc_rw * w_a   where w_a ~ N(0,I)
  b_g[k+1] = b_g[k] + sqrt(dt) * sigma_gyro_rw * w_g  where w_g ~ N(0,I)

White noise (per step):
  eta_a ~ N(0, sigma_acc^2 * I / dt)   [velocity random walk, ARW]
  eta_g ~ N(0, sigma_gyro^2 * I / dt)  [angle random walk, ARW]

Error growth: bias grows as sqrt(t) (random walk), velocity error from accel
integrates to linear in t, position error integrates to quadratic in t.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class IMUModel:
    """IMU sensor model with bias, white noise, and random walk.

    Parameters
    ----------
    sigma_acc : float
        Accelerometer white noise std (m/s^2/sqrt(Hz) or similar scale).
    sigma_gyro : float
        Gyroscope white noise std (rad/s/sqrt(Hz) or similar scale).
    sigma_acc_rw : float
        Accelerometer bias random walk std (m/s^3/sqrt(Hz)).
    sigma_gyro_rw : float
        Gyroscope bias random walk std (rad/s^2/sqrt(Hz)).
    bias_acc_init : np.ndarray or None
        Initial accelerometer bias (3,). Default zeros.
    bias_gyro_init : np.ndarray or None
        Initial gyroscope bias (3,). Default zeros.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma_acc: float = 0.1,
        sigma_gyro: float = 0.01,
        sigma_acc_rw: float = 1e-4,
        sigma_gyro_rw: float = 1e-5,
        bias_acc_init: Optional[np.ndarray] = None,
        bias_gyro_init: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro
        self.sigma_acc_rw = sigma_acc_rw
        self.sigma_gyro_rw = sigma_gyro_rw
        self.bias_acc = (
            np.asarray(bias_acc_init, dtype=float) if bias_acc_init is not None else np.zeros(3)
        )
        self.bias_gyro = (
            np.asarray(bias_gyro_init, dtype=float) if bias_gyro_init is not None else np.zeros(3)
        )
        self._rng = np.random.default_rng(seed)

    def generate_measurements(
        self,
        truth_acc: np.ndarray,
        truth_gyro: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate noisy IMU measurements from ground truth.

        Parameters
        ----------
        truth_acc : np.ndarray
            Ground truth acceleration (N, 3) in m/s^2.
        truth_gyro : np.ndarray
            Ground truth angular velocity (N, 3) in rad/s.
        dt : float
            Time step in seconds.

        Returns
        -------
        acc_meas : np.ndarray
            Noisy accelerometer measurements (N, 3).
        gyro_meas : np.ndarray
            Noisy gyroscope measurements (N, 3).
        """
        n = len(truth_acc)
        truth_acc = np.atleast_2d(truth_acc)
        truth_gyro = np.atleast_2d(truth_gyro)
        if truth_acc.shape[0] != truth_gyro.shape[0]:
            raise ValueError("truth_acc and truth_gyro must have same length")

        acc_meas = np.zeros_like(truth_acc)
        gyro_meas = np.zeros_like(truth_gyro)

        for k in range(n):
            # White noise (scaled by 1/sqrt(dt) for discrete-time ARW)
            eta_a = self._rng.normal(0, self.sigma_acc / np.sqrt(dt), 3)
            eta_g = self._rng.normal(0, self.sigma_gyro / np.sqrt(dt), 3)

            # Bias random walk update: b[k+1] = b[k] + sqrt(dt) * sigma_rw * w
            self.bias_acc += np.sqrt(dt) * self.sigma_acc_rw * self._rng.standard_normal(3)
            self.bias_gyro += np.sqrt(dt) * self.sigma_gyro_rw * self._rng.standard_normal(3)

            # Measurement: meas = truth + bias + white_noise
            acc_meas[k] = truth_acc[k] + self.bias_acc + eta_a
            gyro_meas[k] = truth_gyro[k] + self.bias_gyro + eta_g

        return acc_meas, gyro_meas

    def reset_bias(
        self,
        bias_acc: Optional[np.ndarray] = None,
        bias_gyro: Optional[np.ndarray] = None,
    ) -> None:
        """Reset bias states (e.g., for new simulation run)."""
        if bias_acc is not None:
            self.bias_acc = np.asarray(bias_acc, dtype=float)
        else:
            self.bias_acc = np.zeros(3)
        if bias_gyro is not None:
            self.bias_gyro = np.asarray(bias_gyro, dtype=float)
        else:
            self.bias_gyro = np.zeros(3)

    def plot_error_growth(
        self,
        duration: float = 100.0,
        dt: float = 0.01,
        n_runs: int = 10,
        seed: Optional[int] = None,
        ax=None,
    ) -> plt.Figure:
        """Visualize bias and integrated error growth over time.

        Uses zero true motion; errors come purely from IMU noise.
        Bias grows as sqrt(t); velocity error grows linearly; position error
        grows quadratically.

        Parameters
        ----------
        duration : float
            Simulation duration in seconds.
        dt : float
            Time step.
        n_runs : int
            Number of Monte Carlo runs for statistics.
        seed : int or None
            Seed for reproducibility.
        ax : matplotlib axes or None
            Axes to plot on; if None, creates new figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        n_steps = int(duration / dt)
        t = np.arange(n_steps) * dt

        bias_acc_norm = np.zeros((n_runs, n_steps))
        bias_gyro_norm = np.zeros((n_runs, n_steps))
        vel_error_norm = np.zeros((n_runs, n_steps))
        pos_error_norm = np.zeros((n_runs, n_steps))

        truth_acc = np.zeros((n_steps, 3))
        truth_gyro = np.zeros((n_steps, 3))

        for run in range(n_runs):
            model = IMUModel(
                sigma_acc=self.sigma_acc,
                sigma_gyro=self.sigma_gyro,
                sigma_acc_rw=self.sigma_acc_rw,
                sigma_gyro_rw=self.sigma_gyro_rw,
                seed=seed if seed is not None else run,
            )
            vel = np.zeros(3)
            pos = np.zeros(3)
            for k in range(n_steps):
                acc_meas, gyro_meas = model.generate_measurements(
                    truth_acc[k : k + 1], truth_gyro[k : k + 1], dt
                )
                bias_acc_norm[run, k] = np.linalg.norm(model.bias_acc)
                bias_gyro_norm[run, k] = np.linalg.norm(model.bias_gyro)
                vel += acc_meas[0] * dt
                pos += vel * dt
                vel_error_norm[run, k] = np.linalg.norm(vel)
                pos_error_norm[run, k] = np.linalg.norm(pos)

        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        else:
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            axes = np.atleast_2d(ax)
            if axes.ndim == 1:
                axes = axes.reshape(1, -1)
        flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Bias acc
        ax0 = flat_axes[0]
        for run in range(n_runs):
            ax0.plot(t, bias_acc_norm[run], alpha=0.5, color="C0")
        ax0.plot(t, np.mean(bias_acc_norm, axis=0), "k-", lw=2, label="Mean")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("||bias_acc|| (m/s²)")
        ax0.set_title("Accelerometer Bias (Random Walk)")
        ax0.legend()
        ax0.grid(True, alpha=0.3)

        # Bias gyro
        ax1 = flat_axes[1]
        for run in range(n_runs):
            ax1.plot(t, bias_gyro_norm[run], alpha=0.5, color="C1")
        ax1.plot(t, np.mean(bias_gyro_norm, axis=0), "k-", lw=2, label="Mean")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("||bias_gyro|| (rad/s)")
        ax1.set_title("Gyroscope Bias (Random Walk)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Velocity error
        ax2 = flat_axes[2]
        for run in range(n_runs):
            ax2.plot(t, vel_error_norm[run], alpha=0.5, color="C2")
        ax2.plot(t, np.mean(vel_error_norm, axis=0), "k-", lw=2, label="Mean")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("||vel_error|| (m/s)")
        ax2.set_title("Integrated Velocity Error (linear growth)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Position error
        ax3 = flat_axes[3]
        for run in range(n_runs):
            ax3.plot(t, pos_error_norm[run], alpha=0.5, color="C3")
        ax3.plot(t, np.mean(pos_error_norm, axis=0), "k-", lw=2, label="Mean")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("||pos_error|| (m)")
        ax3.set_title("Integrated Position Error (quadratic growth)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
