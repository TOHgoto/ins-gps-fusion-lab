"""Basic INS-GPS fusion experiment.

Generates truth trajectory, simulates IMU and GPS, runs Kalman filter,
and plots truth vs estimate.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ins_gps_fusion_lab.filters.kalman_filter import KalmanFilter
from ins_gps_fusion_lab.simulation.gps_model import GPSModel
from ins_gps_fusion_lab.simulation.imu_model import IMUModel
from ins_gps_fusion_lab.simulation.state_space import compute_B, compute_F, compute_H, compute_Q
from ins_gps_fusion_lab.simulation.trajectory_generator import TrajectoryGenerator


def run_experiment(
    duration: float = 10.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Run basic fusion loop and plot results.

    Parameters
    ----------
    duration : float
        Simulation duration in seconds.
    dt : float
        IMU time step.
    gps_rate : int
        GPS measurement every gps_rate IMU steps.
    seed : int
        Random seed.
    save_path : str or None
        If set, save figure to this path.
    """
    # Generate truth trajectory
    gen = TrajectoryGenerator(dt=dt)
    positions, velocities, accelerations = gen.straight_line(
        duration=duration,
        velocity=np.array([2.0, 0.5, 0.0]),
        start_pos=np.zeros(3),
    )
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    # Sensors
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
    nis_list = []

    # Main loop
    for k in range(1, n_steps):
        acc_meas, _ = imu.generate_measurements(accelerations[k - 1 : k], np.zeros((1, 3)), dt)
        kf.predict(u=acc_meas[0])

        if k % gps_rate == 0:
            pos_meas, valid = gps.generate_measurement(positions[k])
            if valid:
                _, nis, _ = kf.update(pos_meas)
                nis_list.append((k, nis))

        est_positions[k] = kf.x[0:3]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(t, positions[:, 0], "b-", label="Truth")
    axes[0, 0].plot(t, est_positions[:, 0], "r--", label="Estimate", alpha=0.8)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].set_title("Position X")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, positions[:, 1], "b-", label="Truth")
    axes[0, 1].plot(t, est_positions[:, 1], "r--", label="Estimate", alpha=0.8)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].set_title("Position Y")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    pos_error = np.linalg.norm(positions - est_positions, axis=1)
    axes[1, 0].plot(t, pos_error, "g-")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Error (m)")
    axes[1, 0].set_title("Position Error Norm")
    axes[1, 0].grid(True, alpha=0.3)

    if nis_list:
        k_vals = [x[0] for x in nis_list]
        nis_vals = [x[1] for x in nis_list]
        axes[1, 1].plot(np.array(k_vals) * dt, nis_vals, "o-", markersize=3)
        axes[1, 1].axhline(y=3.0, color="r", linestyle="--", label="dof=3 mean")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("NIS")
        axes[1, 1].set_title("NIS at GPS Updates")
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    run_experiment(duration=10.0, gps_rate=10)
