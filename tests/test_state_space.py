"""Unit tests for state space matrices."""

import numpy as np

from ins_gps_fusion_lab.simulation.state_space import (
    compute_F,
    compute_G,
    compute_H,
    compute_Q,
)


def test_F_shape():
    """F is 12x12."""
    F = compute_F(0.01)
    assert F.shape == (12, 12)


def test_F_identity_blocks():
    """Diagonal blocks are identity."""
    F = compute_F(0.01)
    assert np.allclose(F[0:3, 0:3], np.eye(3))
    assert np.allclose(F[3:6, 3:6], np.eye(3))
    assert np.allclose(F[6:9, 6:9], np.eye(3))
    assert np.allclose(F[9:12, 9:12], np.eye(3))


def test_F_p_v_coupling():
    """p += v*dt, v += -b_a*dt."""
    dt = 0.1
    F = compute_F(dt)
    assert np.allclose(F[0:3, 3:6], dt * np.eye(3))
    assert np.allclose(F[3:6, 6:9], -dt * np.eye(3))


def test_H_position_only():
    """H observes position (first 3 states)."""
    H = compute_H()
    assert H.shape == (3, 12)
    assert np.allclose(H[:, 0:3], np.eye(3))
    assert np.allclose(H[:, 3:12], 0)


def test_Q_positive_semidefinite():
    """Q is symmetric positive semi-definite (position has no direct noise)."""
    Q = compute_Q(0.01)
    assert Q.shape == (12, 12)
    assert np.allclose(Q, Q.T)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= -1e-10)
    assert np.any(eigvals > 1e-10)


def test_G_shape():
    """G is 12x9."""
    G = compute_G(0.01)
    assert G.shape == (12, 9)
