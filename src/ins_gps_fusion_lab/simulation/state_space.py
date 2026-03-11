"""State space matrices for INS-GPS linear Kalman filter.

State: x = [p_x, p_y, p_z, v_x, v_y, v_z, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
- p: position (3)
- v: velocity (3)
- b_a: accelerometer bias (3)
- b_g: gyroscope bias (3)
"""

import numpy as np


def compute_F(dt: float) -> np.ndarray:
    """Compute discrete transition matrix F (12x12).

    x_{k+1} = F @ x_k + B @ u_k (with u = acceleration from IMU)
    For state propagation without control: F @ x gives the deterministic part.

    p_new = p + v*dt
    v_new = v - b_a*dt  (when u=0; with u we add B@u)
    b_a_new = b_a
    b_g_new = b_g
    """
    F = np.eye(12)
    I3 = np.eye(3)
    # p += v * dt
    F[0:3, 3:6] = dt * I3
    # v += -b_a * dt
    F[3:6, 6:9] = -dt * I3
    return F


def compute_B(dt: float) -> np.ndarray:
    """Compute control input matrix B (12x3) for acceleration input u = a_meas.

    v_new = v + a_meas*dt - b_a*dt, so B maps a_meas to velocity update.
    """
    B = np.zeros((12, 3))
    B[3:6, 0:3] = dt * np.eye(3)
    return B


def compute_G(dt: float) -> np.ndarray:
    """Compute process noise input matrix G (12x9).

    w = [w_a(3), w_ba(3), w_bg(3)]: accel noise, accel bias RW, gyro bias RW.
    w_a affects velocity; w_ba affects b_a; w_bg affects b_g.
    """
    G = np.zeros((12, 9))
    I3 = np.eye(3)
    G[3:6, 0:3] = dt * I3  # w_a -> velocity
    G[6:9, 3:6] = I3  # w_ba -> b_a
    G[9:12, 6:9] = I3  # w_bg -> b_g
    return G


def compute_H() -> np.ndarray:
    """Compute measurement matrix H (3x12) for GPS position.

    z = H @ x + v,  H observes position only.
    """
    H = np.zeros((3, 12))
    H[0:3, 0:3] = np.eye(3)
    return H


def compute_Q(
    dt: float,
    sigma_acc: float = 0.1,
    sigma_gyro: float = 0.01,
    sigma_acc_rw: float = 1e-4,
    sigma_gyro_rw: float = 1e-5,
) -> np.ndarray:
    """Compute process noise covariance Q (12x12).

    Q = G @ Qw @ G.T where Qw = diag(Q_a, Q_ba, Q_bg):
    - Q_a: accel white noise -> velocity RW, var = sigma_acc^2 * dt per axis
    - Q_ba: accel bias RW, var = sigma_acc_rw^2 * dt per axis
    - Q_bg: gyro bias RW, var = sigma_gyro_rw^2 * dt per axis
    """
    G = compute_G(dt)
    Qw = np.zeros((9, 9))
    Qw[0:3, 0:3] = (sigma_acc**2) * dt * np.eye(3)
    Qw[3:6, 3:6] = (sigma_acc_rw**2) * dt * np.eye(3)
    Qw[6:9, 6:9] = (sigma_gyro_rw**2) * dt * np.eye(3)
    return G @ Qw @ G.T
