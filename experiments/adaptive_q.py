"""Experiment: Adaptive Q vs fixed Q.

When NIS exceeds threshold for N consecutive steps, scale Q by factor.
Decay back when NIS normalizes. Compare fixed Q vs adaptive Q on Q-mismatch scenario.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from simulation.trajectory_generator import TrajectoryGenerator
from simulation.imu_model import IMUModel
from simulation.gps_model import GPSModel
from simulation.state_space import compute_F, compute_B, compute_H, compute_Q
from filters.kalman_filter import KalmanFilter
from visualization.plot_nis import plot_nis


def run_with_adaptive_q(
    positions,
    velocities,
    accelerations,
    dt,
    gps_rate,
    Q_base,
    gps,
    imu,
    Q_scale_factor: float = 2.0,
    n_consecutive: int = 3,
    alpha: float = 0.05,
    dof: int = 3,
):
    """Run filter with adaptive Q: scale Q when NIS exceeds threshold for n_consecutive steps."""
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    F = compute_F(dt)
    B = compute_B(dt)
    H = compute_H()
    R = np.eye(3) * 1.0

    x0 = np.zeros(12)
    x0[0:3] = positions[0]
    x0[3:6] = velocities[0]
    P0 = np.eye(12) * 0.1

    kf = KalmanFilter(x=x0, P=P0, F=F, Q=Q_base, H=H, R=R, B=B)
    threshold = stats.chi2.ppf(1 - alpha, dof)

    est_positions = np.zeros((n_steps, 3))
    est_positions[0] = x0[0:3]
    nis_list = []
    q_scale_history = []
    consecutive_high = 0
    current_q_scale = 1.0

    for k in range(1, n_steps):
        acc_meas, _ = imu.generate_measurements(
            accelerations[k - 1 : k], np.zeros((1, 3)), dt
        )
        kf.predict(u=acc_meas[0], Q=Q_base * current_q_scale)

        if k % gps_rate == 0:
            pos_meas, valid = gps.generate_measurement(positions[k])
            if valid:
                _, nis, _ = kf.update(pos_meas)
                nis_list.append((k * dt, nis))

                if nis > threshold:
                    consecutive_high += 1
                    if consecutive_high >= n_consecutive:
                        current_q_scale = min(current_q_scale * Q_scale_factor, 100.0)
                else:
                    consecutive_high = 0
                    current_q_scale = max(current_q_scale / 1.1, 1.0)

        q_scale_history.append(current_q_scale)
        est_positions[k] = kf.x[0:3]

    pos_error = np.linalg.norm(positions - est_positions, axis=1)
    return est_positions, nis_list, pos_error, t, q_scale_history


def run_experiment(
    duration: float = 30.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Compare fixed Q vs adaptive Q on Q-mismatch scenario."""
    gen = TrajectoryGenerator(dt=dt)
    positions, velocities, accelerations = gen.straight_line(
        duration=duration,
        velocity=np.array([2.0, 0.5, 0.0]),
        start_pos=np.zeros(3),
    )
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    imu = IMUModel(
        sigma_acc=0.1,
        sigma_gyro=0.01,
        sigma_acc_rw=1e-3,
        sigma_gyro_rw=1e-4,
        seed=seed,
    )
    gps = GPSModel(sigma_pos=1.0, outlier_prob=0.0, dropout_prob=0.0, seed=seed)

    Q_small = compute_Q(dt, sigma_acc=0.1, sigma_acc_rw=1e-6, sigma_gyro_rw=1e-7)

    # Fixed Q run
    F = compute_F(dt)
    B = compute_B(dt)
    H = compute_H()
    R = np.eye(3) * 1.0
    x0 = np.zeros(12)
    x0[0:3] = positions[0]
    x0[3:6] = velocities[0]
    P0 = np.eye(12) * 0.1

    kf_fixed = KalmanFilter(x=x0.copy(), P=P0.copy(), F=F, Q=Q_small, H=H, R=R, B=B)
    imu_fixed = IMUModel(sigma_acc=0.1, sigma_gyro=0.01, sigma_acc_rw=1e-3, sigma_gyro_rw=1e-4, seed=seed)
    gps_fixed = GPSModel(sigma_pos=1.0, outlier_prob=0.0, dropout_prob=0.0, seed=seed)

    est_fixed = np.zeros((n_steps, 3))
    est_fixed[0] = x0[0:3]
    nis_fixed = []

    for k in range(1, n_steps):
        acc_meas, _ = imu_fixed.generate_measurements(
            accelerations[k - 1 : k], np.zeros((1, 3)), dt
        )
        kf_fixed.predict(u=acc_meas[0])
        if k % gps_rate == 0:
            pos_meas, valid = gps_fixed.generate_measurement(positions[k])
            if valid:
                _, nis, _ = kf_fixed.update(pos_meas)
                nis_fixed.append((k * dt, nis))
        est_fixed[k] = kf_fixed.x[0:3]

    err_fixed = np.linalg.norm(positions - est_fixed, axis=1)

    # Adaptive Q run
    est_adapt, nis_adapt, err_adapt, _, q_scale_hist = run_with_adaptive_q(
        positions, velocities, accelerations,
        dt, gps_rate, Q_small,
        gps, imu,
    )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(t, positions[:, 0], "b-", label="Truth")
    axes[0, 0].plot(t, est_fixed[:, 0], "r--", alpha=0.8, label="Fixed Q")
    axes[0, 0].plot(t, est_adapt[:, 0], "g-.", alpha=0.8, label="Adaptive Q")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].set_title("Position X")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, err_fixed, "r-", label="Fixed Q")
    axes[0, 1].plot(t, err_adapt, "g-", label="Adaptive Q")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title("Position Error (adaptive can reduce drift)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if nis_fixed:
        t_nis = np.array([x[0] for x in nis_fixed])
        plot_nis(np.array([x[1] for x in nis_fixed]), dof=3, t=t_nis, ax=axes[1, 0])
        axes[1, 0].set_title("NIS (Fixed Q)")

    axes[1, 1].plot(t[1:], q_scale_hist, "g-")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Q scale factor")
    axes[1, 1].set_title("Adaptive Q Scale Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Adaptive Q vs Fixed Q (Q-mismatch scenario)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    run_experiment()
