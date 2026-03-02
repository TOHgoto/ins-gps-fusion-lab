"""Unit tests for GPS model."""

import numpy as np
import pytest

from simulation.gps_model import GPSModel


def test_gps_reproducibility():
    """Same seed produces same measurements."""
    true_pos = np.array([10.0, 20.0, 0.0])
    gps1 = GPSModel(seed=42)
    pos1, valid1 = gps1.generate_measurement(true_pos)
    gps2 = GPSModel(seed=42)
    pos2, valid2 = gps2.generate_measurement(true_pos)
    np.testing.assert_array_almost_equal(pos1, pos2)
    assert valid1 == valid2


def test_gps_output_shape():
    """Single measurement returns (3,) position and bool."""
    gps = GPSModel(seed=0)
    pos, valid = gps.generate_measurement(np.array([1.0, 2.0, 3.0]))
    assert pos.shape == (3,)
    assert isinstance(valid, bool)


def test_gps_batch_output_shape():
    """Batch measurements return (N, 3) and (N,) valid."""
    n = 20
    true_positions = np.random.randn(n, 3) * 100
    gps = GPSModel(seed=0)
    pos, valid = gps.generate_measurements(true_positions)
    assert pos.shape == (n, 3)
    assert valid.shape == (n,)
    assert valid.dtype == bool


def test_gps_zero_noise_gives_truth():
    """With zero sigma, no outlier, no dropout: measurement equals truth."""
    gps = GPSModel(sigma_pos=0.0, outlier_prob=0.0, dropout_prob=0.0, seed=0)
    true_pos = np.array([1.0, 2.0, 3.0])
    pos, valid = gps.generate_measurement(true_pos)
    np.testing.assert_array_almost_equal(pos, true_pos)
    assert valid is True


def test_gps_dropout_produces_invalid():
    """With dropout_prob=1, all measurements invalid."""
    gps = GPSModel(dropout_prob=1.0, seed=0)
    for _ in range(10):
        _, valid = gps.generate_measurement(np.array([1.0, 2.0, 3.0]))
        assert valid is False


def test_gps_outlier_increases_error():
    """With outlier_prob=1, errors are larger than normal noise."""
    gps_normal = GPSModel(sigma_pos=1.0, outlier_prob=0.0, seed=0)
    gps_outlier = GPSModel(
        sigma_pos=1.0, outlier_prob=1.0, outlier_scale=10.0, seed=0
    )
    true_pos = np.array([0.0, 0.0, 0.0])
    errors_normal = []
    errors_outlier = []
    for _ in range(50):
        pos_n, _ = gps_normal.generate_measurement(true_pos)
        pos_o, _ = gps_outlier.generate_measurement(true_pos)
        errors_normal.append(np.linalg.norm(pos_n))
        errors_outlier.append(np.linalg.norm(pos_o))
    assert np.mean(errors_outlier) > np.mean(errors_normal)


def test_gps_compute_residual():
    """Residual = measured - predicted."""
    gps = GPSModel(seed=0)
    measured = np.array([11.0, 21.0, 1.0])
    predicted = np.array([10.0, 20.0, 0.0])
    residual = gps.compute_residual(measured, predicted)
    np.testing.assert_array_almost_equal(residual, np.array([1.0, 1.0, 1.0]))


def test_gps_sigma_per_axis():
    """Per-axis sigma_pos works."""
    gps = GPSModel(sigma_pos=np.array([0.1, 1.0, 10.0]), seed=0)
    true_pos = np.zeros(3)
    pos, _ = gps.generate_measurement(true_pos)
    # Variance of third component should be ~100x first
    assert abs(pos[2]) > abs(pos[0]) or abs(pos[2]) > abs(pos[1])
