"""Unit tests for Kalman filter."""

import numpy as np

from ins_gps_fusion_lab.filters.kalman_filter import KalmanFilter
from ins_gps_fusion_lab.simulation.state_space import (
    compute_B,
    compute_F,
    compute_H,
    compute_Q,
)


def _make_filter():
    """Create a minimal 12-state INS-GPS filter."""
    n = 12
    dt = 0.01
    x = np.zeros(n)
    P = np.eye(n) * 0.1
    F = compute_F(dt)
    Q = compute_Q(dt, sigma_acc=0.1, sigma_acc_rw=1e-4, sigma_gyro_rw=1e-5)
    H = compute_H()
    R = np.eye(3) * 1.0
    B = compute_B(dt)
    return KalmanFilter(x=x, P=P, F=F, Q=Q, H=H, R=R, B=B)


def test_predict_only_p_grows():
    """Prediction without update: P grows (trace increases)."""
    kf = _make_filter()
    trace_before = np.trace(kf.P)
    for _ in range(10):
        kf.predict(u=np.zeros(3))
    trace_after = np.trace(kf.P)
    assert trace_after > trace_before


def test_update_shrinks_p():
    """Update with measurement: P shrinks."""
    kf = _make_filter()
    kf.predict(u=np.zeros(3))
    trace_before = np.trace(kf.P)
    z = kf.H @ kf.x  # Perfect measurement (no noise)
    kf.update(z)
    trace_after = np.trace(kf.P)
    assert trace_after < trace_before


def test_nis_chi_squared():
    """NIS should be chi-squared with dof=3."""
    kf = _make_filter()
    nis_list = []
    np.random.seed(42)
    for _ in range(200):
        kf.predict(u=np.zeros(3))
        z = kf.H @ kf.x + np.random.randn(3) * np.sqrt(np.diag(kf.R))
        _, nis, _ = kf.update(z)
        nis_list.append(nis)
    # Mean of chi-squared(dof) = dof
    mean_nis = np.mean(nis_list)
    assert 1.5 < mean_nis < 5.0  # dof=3, expect ~3


def test_reproducibility():
    """Same inputs give same outputs."""
    kf1 = _make_filter()
    kf2 = _make_filter()
    np.random.seed(0)
    u = np.random.randn(3) * 0.1
    z = np.random.randn(3) * 2.0
    kf1.predict(u=u)
    kf2.predict(u=u)
    _, nis1, _ = kf1.update(z)
    _, nis2, _ = kf2.update(z)
    np.testing.assert_array_almost_equal(kf1.x, kf2.x)
    np.testing.assert_array_almost_equal(kf1.P, kf2.P)
    assert abs(nis1 - nis2) < 1e-10


def test_update_with_correct_measurement():
    """Update with true position: state moves toward measurement."""
    kf = _make_filter()
    true_pos = np.array([10.0, 20.0, 0.0])
    kf.x[0:3] = 0.0  # Wrong initial position
    kf.x[3:6] = np.array([1.0, 0.0, 0.0])  # Some velocity
    kf.predict(u=np.zeros(3))
    pred_pos = kf.x[0:3].copy()
    z = true_pos.copy()
    kf.update(z)
    # Position should move toward z (closer than before)
    dist_before = np.linalg.norm(pred_pos - true_pos)
    dist_after = np.linalg.norm(kf.x[0:3] - true_pos)
    assert dist_after < dist_before


def test_innovation_gating_rejects_outlier():
    """Gating rejects measurement when NIS exceeds threshold."""
    kf = _make_filter()
    kf.predict(u=np.zeros(3))
    x_before = kf.x.copy()
    P_before = kf.P.copy()
    # Huge outlier: z far from predicted
    z_outlier = kf.H @ kf.x + np.array([100, 100, 100])
    _, nis, rejected = kf.update(z_outlier, gate_alpha=0.05)
    assert rejected is True
    np.testing.assert_array_almost_equal(kf.x, x_before)
    np.testing.assert_array_almost_equal(kf.P, P_before)
