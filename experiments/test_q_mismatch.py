"""Experiment: Q mismatch consequences.

When actual IMU bias is large but Q is set very small:
- Verify covariance evolution (P underestimates error)
- NIS spikes / filter inconsistency
- Trajectory drift
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from ins_gps_fusion_lab.simulation.trajectory_generator import TrajectoryGenerator
from ins_gps_fusion_lab.simulation.imu_model import IMUModel
from ins_gps_fusion_lab.simulation.gps_model import GPSModel
from ins_gps_fusion_lab.simulation.state_space import compute_F, compute_B, compute_H, compute_Q
from ins_gps_fusion_lab.filters.kalman_filter import KalmanFilter
from ins_gps_fusion_lab.visualization.plot_nis import plot_nis
from ins_gps_fusion_lab.visualization.plot_covariance import plot_covariance


def run_experiment(
    duration: float = 30.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Run Q mismatch experiment.

    True IMU: sigma_acc_rw=1e-3 (large bias growth)
    Filter Q: sigma_acc_rw=1e-6 (assumes small bias)
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

    # IMU with LARGE bias (true process)
    imu_true = IMUModel(
        sigma_acc=0.1,
        sigma_gyro=0.01,
        sigma_acc_rw=1e-3,   # Large
        sigma_gyro_rw=1e-4,
        seed=seed,
    )
    gps = GPSModel(sigma_pos=1.0, outlier_prob=0.0, dropout_prob=0.0, seed=seed)

    # Filter uses SMALL Q (mismatch)
    F = compute_F(dt)
    B = compute_B(dt)
    Q_small = compute_Q(dt, sigma_acc=0.1, sigma_acc_rw=1e-6, sigma_gyro_rw=1e-7)
    H = compute_H()
    R = np.eye(3) * 1.0

    x0 = np.zeros(12)
    x0[0:3] = positions[0]
    x0[3:6] = velocities[0]
    P0 = np.eye(12) * 0.1

    kf = KalmanFilter(x=x0, P=P0, F=F, Q=Q_small, H=H, R=R, B=B)

    est_positions = np.zeros((n_steps, 3))
    est_positions[0] = x0[0:3]
    P_history = [P0.copy()]
    nis_list = []
    pos_errors = []

    for k in range(1, n_steps):
        acc_meas, _ = imu_true.generate_measurements(
            accelerations[k - 1 : k], np.zeros((1, 3)), dt
        )
        kf.predict(u=acc_meas[0])

        if k % gps_rate == 0:
            pos_meas, valid = gps.generate_measurement(positions[k])
            if valid:
                _, nis, _ = kf.update(pos_meas)
                nis_list.append((k * dt, nis))

        est_positions[k] = kf.x[0:3]
        P_history.append(kf.P.copy())
        pos_errors.append(np.linalg.norm(positions[k] - est_positions[k]))

    P_history = np.array(P_history)
    t_P = np.arange(len(P_history)) * dt

    # Analysis: P[0,0] vs actual position error variance proxy
    P_pos_diag = np.array([np.trace(P[0:3, 0:3]) for P in P_history])
    pos_error_arr = np.array(pos_errors)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Trajectory
    axes[0, 0].plot(t, positions[:, 0], "b-", label="Truth")
    axes[0, 0].plot(t, est_positions[:, 0], "r--", label="Estimate", alpha=0.8)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].set_title("Position X (drift from Q mismatch)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position error vs sqrt(P) - P underestimates when Q too small
    axes[0, 1].plot(t[1:], pos_error_arr, "g-", label="Actual error")
    axes[0, 1].plot(t_P, np.sqrt(P_pos_diag), "r--", label="sqrt(trace(P_pos))")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].set_title("P underestimates error (filter inconsistent)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # NIS - should spike
    if nis_list:
        t_nis = np.array([x[0] for x in nis_list])
        nis_vals = np.array([x[1] for x in nis_list])
        plot_nis(nis_vals, dof=3, alpha=0.05, t=t_nis, ax=axes[1, 0])
        axes[1, 0].set_title("NIS (spikes indicate inconsistency)")

    # P trace
    plot_covariance(P_history, t=t_P, ax=axes[1, 1], indices=[0, 1, 2])
    axes[1, 1].set_title("Covariance (stays small despite large error)")

    plt.suptitle("Q Mismatch: True sigma_acc_rw=1e-3, Filter uses 1e-6")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    run_experiment()
