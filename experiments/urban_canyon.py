"""Experiment: Urban canyon GPS anomalies.

Simulate urban canyon with GPS outliers:
- Compare: no gating vs gating vs increased R
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ins_gps_fusion_lab.filters.kalman_filter import KalmanFilter
from ins_gps_fusion_lab.simulation.gps_model import GPSModel
from ins_gps_fusion_lab.simulation.imu_model import IMUModel
from ins_gps_fusion_lab.simulation.state_space import compute_B, compute_F, compute_H, compute_Q
from ins_gps_fusion_lab.simulation.trajectory_generator import TrajectoryGenerator
from ins_gps_fusion_lab.visualization.animation_backends import UrbanCanyonAnimationBackend
from ins_gps_fusion_lab.visualization.animation_manager import AnimationManager
from ins_gps_fusion_lab.visualization.plot_nis import plot_nis


def _run_single(
    positions,
    velocities,
    accelerations,
    dt,
    gps_rate,
    use_gating: bool,
    gate_alpha: Optional[float],
    R_scale: float,
    gps,
    imu,
):
    """Run one fusion run with given settings."""
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    F = compute_F(dt)
    B = compute_B(dt)
    Q = compute_Q(dt, sigma_acc=0.1, sigma_acc_rw=1e-4, sigma_gyro_rw=1e-5)
    H = compute_H()
    R = np.eye(3) * 1.0 * R_scale

    x0 = np.zeros(12)
    x0[0:3] = positions[0]
    x0[3:6] = velocities[0]
    P0 = np.eye(12) * 0.1

    kf = KalmanFilter(x=x0, P=P0, F=F, Q=Q, H=H, R=R, B=B)

    est_positions = np.zeros((n_steps, 3))
    est_positions[0] = x0[0:3]
    nis_list = []

    for k in range(1, n_steps):
        acc_meas, _ = imu.generate_measurements(accelerations[k - 1 : k], np.zeros((1, 3)), dt)
        kf.predict(u=acc_meas[0])

        if k % gps_rate == 0:
            pos_meas, valid = gps.generate_measurement(positions[k])
            if valid:
                if use_gating and gate_alpha is not None:
                    _, nis, rejected = kf.update(pos_meas, gate_alpha=gate_alpha)
                    if not rejected:
                        nis_list.append((k * dt, nis))
                else:
                    _, nis, _ = kf.update(pos_meas)
                    nis_list.append((k * dt, nis))

        est_positions[k] = kf.x[0:3]

    pos_error = np.linalg.norm(positions - est_positions, axis=1)
    return est_positions, nis_list, pos_error, t


def simulate_urban_canyon(
    duration: float = 20.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
):
    """Run urban canyon simulation and return comparison time-series data."""
    gps = GPSModel(
        sigma_pos=1.0,
        outlier_prob=0.1,
        outlier_scale=20.0,
        dropout_prob=0.0,
        seed=seed,
    )
    imu = IMUModel(
        sigma_acc=0.1,
        sigma_gyro=0.01,
        sigma_acc_rw=1e-4,
        sigma_gyro_rw=1e-5,
        seed=seed,
    )

    gen = TrajectoryGenerator(dt=dt)
    positions, velocities, accelerations = gen.straight_line(
        duration=duration,
        velocity=np.array([2.0, 0.5, 0.0]),
        start_pos=np.zeros(3),
    )

    # Run 1: No gating
    est1, nis1, err1, t = _run_single(
        positions,
        velocities,
        accelerations,
        dt,
        gps_rate,
        use_gating=False,
        gate_alpha=None,
        R_scale=1.0,
        gps=gps,
        imu=imu,
    )

    # Run 2: With gating (need fresh gps for same randomness - use same seed)
    gps2 = GPSModel(
        sigma_pos=1.0, outlier_prob=0.1, outlier_scale=20.0, dropout_prob=0.0, seed=seed
    )
    imu2 = IMUModel(
        sigma_acc=0.1, sigma_gyro=0.01, sigma_acc_rw=1e-4, sigma_gyro_rw=1e-5, seed=seed
    )
    est2, nis2, err2, _ = _run_single(
        positions,
        velocities,
        accelerations,
        dt,
        gps_rate,
        use_gating=True,
        gate_alpha=0.05,
        R_scale=1.0,
        gps=gps2,
        imu=imu2,
    )

    # Run 3: Increased R (no gating)
    gps3 = GPSModel(
        sigma_pos=1.0, outlier_prob=0.1, outlier_scale=20.0, dropout_prob=0.0, seed=seed
    )
    imu3 = IMUModel(
        sigma_acc=0.1, sigma_gyro=0.01, sigma_acc_rw=1e-4, sigma_gyro_rw=1e-5, seed=seed
    )
    est3, nis3, err3, _ = _run_single(
        positions,
        velocities,
        accelerations,
        dt,
        gps_rate,
        use_gating=False,
        gate_alpha=None,
        R_scale=10.0,
        gps=gps3,
        imu=imu3,
    )

    return {
        "t": t,
        "positions": positions,
        "est1": est1,
        "est2": est2,
        "est3": est3,
        "err1": err1,
        "err2": err2,
        "err3": err3,
        "nis1_times": np.array([x[0] for x in nis1]) if nis1 else np.array([]),
        "nis1_values": np.array([x[1] for x in nis1]) if nis1 else np.array([]),
        "nis2_times": np.array([x[0] for x in nis2]) if nis2 else np.array([]),
        "nis2_values": np.array([x[1] for x in nis2]) if nis2 else np.array([]),
    }


def _plot_static(data: dict, save_path: Optional[str] = None):
    """Render the original static dashboard from simulation data."""
    t = data["t"]
    positions = data["positions"]
    est1 = data["est1"]
    est2 = data["est2"]
    est3 = data["est3"]
    err1 = data["err1"]
    err2 = data["err2"]
    err3 = data["err3"]
    nis1_times = data["nis1_times"]
    nis1_values = data["nis1_values"]
    nis2_times = data["nis2_times"]
    nis2_values = data["nis2_values"]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(t, positions[:, 0], "b-", label="Truth")
    axes[0, 0].plot(t, est1[:, 0], "r--", alpha=0.8, label="No gating")
    axes[0, 0].plot(t, est2[:, 0], "g-.", alpha=0.8, label="Gating")
    axes[0, 0].plot(t, est3[:, 0], "m:", alpha=0.8, label="R x10")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x (m)")
    axes[0, 0].set_title("Position X")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, err1, "r-", label="No gating")
    axes[0, 1].plot(t, err2, "g-", label="Gating")
    axes[0, 1].plot(t, err3, "m-", label="R x10")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title("Position Error Norm")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if len(nis1_values):
        plot_nis(nis1_values, dof=3, t=nis1_times, ax=axes[1, 0])
        axes[1, 0].set_title("NIS (No gating)")

    if len(nis2_values):
        axes[1, 1].plot(nis2_times, nis2_values, "go-", markersize=3)
        axes[1, 1].axhline(y=7.81, color="r", linestyle="--", label="chi2(0.95,3)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("NIS")
        axes[1, 1].set_title("NIS (With gating - outliers rejected)")
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Urban Canyon: outlier_prob=0.1, outlier_scale=20")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


def run_experiment(
    duration: float = 20.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Run urban canyon experiment and produce static plots."""
    data = simulate_urban_canyon(duration=duration, dt=dt, gps_rate=gps_rate, seed=seed)
    return _plot_static(data, save_path=save_path)


def run_animation(
    duration: float = 20.0,
    dt: float = 0.01,
    gps_rate: int = 10,
    seed: int = 42,
    fps: int = 20,
    frame_stride: int = 10,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Run urban canyon experiment and render animated comparison playback."""
    data = simulate_urban_canyon(duration=duration, dt=dt, gps_rate=gps_rate, seed=seed)

    step = max(1, int(frame_stride))
    frame_indices = list(range(0, len(data["t"]), step))
    if frame_indices[-1] != len(data["t"]) - 1:
        frame_indices.append(len(data["t"]) - 1)

    backend = UrbanCanyonAnimationBackend(data)
    manager = AnimationManager(backend=backend, frame_indices=frame_indices, fps=fps)

    if output_path:
        manager.save(output_path)
    if show:
        manager.show()
    return manager


if __name__ == "__main__":
    run_experiment()
