"""Unit tests for IMU model."""

import numpy as np
import pytest

from simulation.imu_model import IMUModel


def test_imu_reproducibility():
    """Same seed produces same measurements."""
    truth_acc = np.zeros((100, 3))
    truth_gyro = np.zeros((100, 3))
    dt = 0.01

    imu1 = IMUModel(seed=42)
    acc1, gyro1 = imu1.generate_measurements(truth_acc, truth_gyro, dt)

    imu2 = IMUModel(seed=42)
    acc2, gyro2 = imu2.generate_measurements(truth_acc, truth_gyro, dt)

    np.testing.assert_array_almost_equal(acc1, acc2)
    np.testing.assert_array_almost_equal(gyro1, gyro2)


def test_imu_output_shape():
    """Output shape matches input."""
    n = 50
    truth_acc = np.random.randn(n, 3) * 0.1
    truth_gyro = np.random.randn(n, 3) * 0.01
    dt = 0.01

    imu = IMUModel(seed=0)
    acc, gyro = imu.generate_measurements(truth_acc, truth_gyro, dt)

    assert acc.shape == (n, 3)
    assert gyro.shape == (n, 3)


def test_imu_bias_grows_with_random_walk():
    """Bias norm grows approximately as sqrt(t) over many steps."""
    n_steps = 5000
    dt = 0.01
    truth_acc = np.zeros((n_steps, 3))
    truth_gyro = np.zeros((n_steps, 3))

    imu = IMUModel(sigma_acc_rw=0.001, sigma_gyro_rw=0.0001, seed=123)
    imu.generate_measurements(truth_acc, truth_gyro, dt)

    # After many steps, bias should be non-negligible
    bias_acc_norm = np.linalg.norm(imu.bias_acc)
    bias_gyro_norm = np.linalg.norm(imu.bias_gyro)
    assert bias_acc_norm > 0.001
    assert bias_gyro_norm > 0.0001


def test_imu_zero_noise_gives_truth():
    """With zero noise, measurements equal truth plus constant bias."""
    truth_acc = np.array([[1.0, 0.0, 0.0]])
    truth_gyro = np.array([[0.1, 0.0, 0.0]])
    dt = 0.01

    imu = IMUModel(
        sigma_acc=0,
        sigma_gyro=0,
        sigma_acc_rw=0,
        sigma_gyro_rw=0,
        bias_acc_init=np.zeros(3),
        bias_gyro_init=np.zeros(3),
        seed=0,
    )
    acc, gyro = imu.generate_measurements(truth_acc, truth_gyro, dt)

    np.testing.assert_array_almost_equal(acc[0], truth_acc[0])
    np.testing.assert_array_almost_equal(gyro[0], truth_gyro[0])


def test_imu_reset_bias():
    """reset_bias resets internal bias state."""
    imu = IMUModel(seed=0)
    truth_acc = np.ones((10, 3)) * 0.1
    truth_gyro = np.ones((10, 3)) * 0.01
    imu.generate_measurements(truth_acc, truth_gyro, 0.01)
    assert np.any(imu.bias_acc != 0) or np.any(imu.bias_gyro != 0)

    imu.reset_bias()
    np.testing.assert_array_almost_equal(imu.bias_acc, np.zeros(3))
    np.testing.assert_array_almost_equal(imu.bias_gyro, np.zeros(3))


def test_imu_plot_error_growth_runs():
    """plot_error_growth runs without error and returns figure."""
    imu = IMUModel(seed=0)
    fig = imu.plot_error_growth(duration=1.0, dt=0.01, n_runs=2)
    assert fig is not None
