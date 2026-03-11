"""GPS observation model for sensor fusion simulation.

Outputs position measurements with:
- Gaussian noise
- Optional random outliers (e.g., urban canyon multipath)
- Optional signal dropout (no measurement)
"""

from typing import Optional, Union

import numpy as np


class GPSModel:
    """GPS position measurement model with noise, outliers, and dropout.

    Parameters
    ----------
    sigma_pos : float or np.ndarray
        Position noise std. Scalar for isotropic (m), or (3,) for per-axis.
    outlier_prob : float
        Probability of an outlier at each measurement (0 to 1).
    outlier_scale : float
        Scale factor for outlier magnitude (multiplier of sigma_pos).
    dropout_prob : float
        Probability of no measurement (signal lost) at each call.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sigma_pos: Union[float, np.ndarray] = 1.0,
        outlier_prob: float = 0.0,
        outlier_scale: float = 10.0,
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.sigma_pos = np.atleast_1d(np.asarray(sigma_pos, dtype=float))
        if self.sigma_pos.size == 1:
            self.sigma_pos = np.full(3, float(self.sigma_pos[0]))
        elif self.sigma_pos.size != 3:
            raise ValueError("sigma_pos must be scalar or length-3")
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale
        self.dropout_prob = dropout_prob
        self._rng = np.random.default_rng(seed)

    def generate_measurement(
        self,
        true_position: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Generate a single GPS position measurement.

        Parameters
        ----------
        true_position : np.ndarray
            Ground truth position (3,) in meters.

        Returns
        -------
        position_meas : np.ndarray
            Noisy position measurement (3,). If dropout, returns last valid
            or zeros (caller should check `valid`).
        valid : bool
            False if dropout occurred (no measurement available).
        """
        true_position = np.asarray(true_position, dtype=float).ravel()
        if true_position.size != 3:
            raise ValueError("true_position must have 3 elements")

        # Dropout
        if self._rng.random() < self.dropout_prob:
            return np.zeros(3), False

        # Gaussian noise
        noise = self._rng.normal(0, self.sigma_pos, 3)

        # Outlier
        if self._rng.random() < self.outlier_prob:
            outlier_noise = self._rng.standard_normal(3) * self.outlier_scale * self.sigma_pos
            noise = noise + outlier_noise

        position_meas = true_position + noise
        return position_meas, True

    def generate_measurements(
        self,
        true_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate GPS measurements for a sequence of positions.

        Parameters
        ----------
        true_positions : np.ndarray
            Ground truth positions (N, 3) in meters.

        Returns
        -------
        position_meas : np.ndarray
            Noisy position measurements (N, 3).
        valid : np.ndarray
            Boolean array (N,) indicating valid measurements.
        """
        true_positions = np.atleast_2d(true_positions)
        n = true_positions.shape[0]
        position_meas = np.zeros((n, 3))
        valid = np.ones(n, dtype=bool)

        for k in range(n):
            position_meas[k], valid[k] = self.generate_measurement(true_positions[k])

        return position_meas, valid

    def compute_residual(
        self,
        measured_position: np.ndarray,
        predicted_position: np.ndarray,
    ) -> np.ndarray:
        """Compute observation residual (innovation) for NIS etc.

        residual = measured_position - predicted_position

        Parameters
        ----------
        measured_position : np.ndarray
            GPS measurement (3,).
        predicted_position : np.ndarray
            Filter's predicted position (3,).

        Returns
        -------
        residual : np.ndarray
            Innovation vector (3,).
        """
        return np.asarray(measured_position) - np.asarray(predicted_position)
