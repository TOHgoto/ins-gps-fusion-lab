"""Experiment: GPS dropout simulation.

Simulate 20s GPS loss in the middle of a 60s run:
- IMU-only prediction during dropout
- Record P growth
- Observe correction when GPS recovers
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from ins_gps_fusion_lab.simulation.trajectory_generator import TrajectoryGenerator
from ins_gps_fusion_lab.simulation.imu_model import IMUModel
from ins_gps_fusion_lab.simulation.gps_model import GPSModel
from ins_gps_fusion_lab.simulation.state_space import compute_F, compute_B, compute_H, compute_Q
from ins_gps_fusion_lab.filters.kalman_filter import KalmanFilter
from ins_gps_fusion_lab.visualization.plot_covariance import plot_covariance


def run_experiment(
    duration: float = 60.0,
    dropout_start: float = 20.0,
    dropout_duration: float = 20.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Run GPS dropout experiment.

    Parameters
    ----------
    duration : float
        Total simulation duration (s).
    dropout_start : float
        Time when GPS dropout begins (s).
    dropout_duration : float
        Duration of GPS dropout (s).
    dt : float
        IMU time step.
    gps_rate : int
        GPS measurement every gps_rate IMU steps.
    seed : int
        Random seed.
    save_path : str or None
        If set, save figure.
    """
    dropout_end = dropout_start + dropout_duration

    # Generate truth trajectory
    gen = TrajectoryGenerator(dt=dt)
    positions, velocities, accelerations = gen.straight_line(
        duration=duration,
        velocity=np.array([2.0, 0.5, 0.0]),
        start_pos=np.zeros(3),
    )
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    # Sensors (no random dropout - we control it explicitly)
    imu = IMUModel(
        sigma_acc=0.1,
        sigma_gyro=0.01,
        sigma_acc_rw=1e-4,
        sigma_gyro_rw=1e-5,
        seed=seed,
    )
    gps = GPSModel(sigma_pos=1.0, outlier_prob=0.0, dropout_prob=0.0, seed=seed)

    # Kalman filter
    F = compute_F(dt)
    B = compute_B(dt)
    Q = compute_Q(dt, sigma_acc=0.1, sigma_acc_rw=1e-4, sigma_gyro_rw=1e-5)
    H = compute_H()
    R = np.eye(3) * 1.0

    x0 = np.zeros(12)
    x0[0:3] = positions[0]
    x0[3:6] = velocities[0]
    P0 = np.eye(12) * 0.1

    kf = KalmanFilter(x=x0, P=P0, F=F, Q=Q, H=H, R=R, B=B)

    # Storage
    est_positions = np.zeros((n_steps, 3))
    est_positions[0] = x0[0:3]
    P_history = [P0.copy()]
    nis_list = []
    first_recovery_k = None
    first_recovery_K_norm = None

    # Main loop
    for k in range(1, n_steps):
        t_k = t[k]
        acc_meas, _ = imu.generate_measurements(
            accelerations[k - 1 : k], np.zeros((1, 3)), dt
        )
        kf.predict(u=acc_meas[0])

        # GPS update only when not in dropout window
        in_dropout = dropout_start <= t_k < dropout_end
        if k % gps_rate == 0 and not in_dropout:
            pos_meas, valid = gps.generate_measurement(positions[k])
            if valid:
                # Capture Kalman gain at first recovery step
                if first_recovery_k is None and t_k >= dropout_end:
                    S = H @ kf.P @ H.T + R
                    K = kf.P @ H.T @ np.linalg.solve(S, np.eye(3))
                    first_recovery_K_norm = np.linalg.norm(K)
                    first_recovery_k = k
                _, nis, _ = kf.update(pos_meas)
                nis_list.append((k, nis))

        est_positions[k] = kf.x[0:3]
        P_history.append(kf.P.copy())

    P_history = np.array(P_history)
    t_P = np.arange(len(P_history)) * dt

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Truth vs estimate
    axes[0, 0].plot(t, positions[:, 0], "b-", label="Truth")
    axes[0, 0].plot(t, est_positions[:, 0], "r--", label="Estimate", alpha=0.8)
    axes[0, 0].axvspan(dropout_start, dropout_end, alpha=0.2, color="gray", label="GPS dropout")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].set_title("Position X")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position error
    pos_error = np.linalg.norm(positions - est_positions, axis=1)
    axes[0, 1].plot(t, pos_error, "g-")
    axes[0, 1].axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title("Position Error (grows during dropout)")
    axes[0, 1].grid(True, alpha=0.3)

    # P trace
    plot_covariance(P_history, t=t_P, ax=axes[1, 0], indices=[0, 1, 2])
    axes[1, 0].axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
    axes[1, 0].set_title("Covariance (P grows during dropout)")

    # NIS
    if nis_list:
        k_vals = np.array([x[0] for x in nis_list]) * dt
        nis_vals = [x[1] for x in nis_list]
        axes[1, 1].plot(k_vals, nis_vals, "o-", markersize=3)
        axes[1, 1].axhline(y=3.0, color="r", linestyle="--", label="dof=3 mean")
        axes[1, 1].axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("NIS")
    axes[1, 1].set_title("NIS at GPS Updates")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if first_recovery_K_norm is not None:
        print(f"First post-recovery Kalman gain norm: {first_recovery_K_norm:.4f}")

    plt.suptitle("GPS Dropout Experiment (20s loss at 20-40s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    run_experiment()
