"""Linear discrete-time Kalman filter.

Implements prediction, update, covariance propagation, Kalman gain,
NIS output, Innovation Gating, and Joseph form for numerical stability.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


class KalmanFilter:
    """Linear Kalman filter for INS-GPS fusion.

    Supports dynamic Q and R. Outputs innovation and NIS for consistency checks.
    """

    def __init__(
        self,
        x: np.ndarray,
        P: np.ndarray,
        F: np.ndarray,
        Q: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        B: Optional[np.ndarray] = None,
    ):
        """Initialize filter with state, covariance, and model matrices.

        Parameters
        ----------
        x : np.ndarray (n,)
            Initial state estimate.
        P : np.ndarray (n, n)
            Initial state covariance.
        F : np.ndarray (n, n)
            State transition matrix.
        Q : np.ndarray (n, n)
            Process noise covariance.
        H : np.ndarray (m, n)
            Measurement matrix.
        R : np.ndarray (m, m)
            Measurement noise covariance.
        B : np.ndarray (n, p) or None
            Control input matrix. If None, no control input.
        """
        self.x = np.asarray(x, dtype=float).ravel()
        self.P = np.asarray(P, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = np.asarray(B, dtype=float) if B is not None else None

        self.n = len(self.x)
        self.m = self.H.shape[0]

        # Ensure P is symmetric
        self.P = 0.5 * (self.P + self.P.T)

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
    ) -> None:
        """Prediction step.

        x_pred = F @ x + B @ u  (if B and u provided)
        P_pred = F @ P @ F.T + Q

        Parameters
        ----------
        u : np.ndarray (p,) or None
            Control input (e.g., acceleration from IMU).
        Q : np.ndarray or None
            Override process noise for this step. Default: use self.Q.
        F : np.ndarray or None
            Override transition matrix. Default: use self.F.
        """
        F_ = F if F is not None else self.F
        Q_ = Q if Q is not None else self.Q

        self.x = F_ @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.asarray(u).ravel()

        self.P = F_ @ self.P @ F_.T + Q_
        self.P = 0.5 * (self.P + self.P.T)

    def update(
        self,
        z: np.ndarray,
        R: Optional[np.ndarray] = None,
        gate_alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """Update step with measurement.

        Parameters
        ----------
        z : np.ndarray (m,)
            Measurement vector.
        R : np.ndarray or None
            Override measurement noise. Default: use self.R.
        gate_alpha : float or None
            If set, reject measurement when NIS > chi2(1-alpha, dof).
            Typical alpha=0.05. None disables gating.

        Returns
        -------
        innovation : np.ndarray (m,)
            y = z - H @ x_pred
        nis : float
            NIS = y.T @ inv(S) @ y (chi-squared with dof = m)
        rejected : bool
            True if measurement was rejected by gating.
        """
        R_ = R if R is not None else self.R
        z = np.asarray(z).ravel()

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R_
        S = 0.5 * (S + S.T)

        # NIS (compute before potential rejection)
        nis = float(y.T @ np.linalg.solve(S, y))

        # Innovation gating
        if gate_alpha is not None:
            threshold = stats.chi2.ppf(1 - gate_alpha, self.m)
            if nis > threshold:
                return y, nis, True

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(self.m))

        # State update
        self.x = self.x + K @ y

        # Joseph form for P (numerically stable)
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_ @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return y, nis, False

    def get_state(self) -> np.ndarray:
        """Return current state estimate."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return current state covariance."""
        return self.P.copy()
