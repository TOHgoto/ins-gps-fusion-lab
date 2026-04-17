"""Matplotlib animation backends for simulation dashboards."""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class AnimationBackend(ABC):
    """Backend interface used by AnimationManager."""

    def __init__(self) -> None:
        self.figure: Figure | None = None

    @abstractmethod
    def setup(self) -> None:
        """Create axes and static artists."""

    @abstractmethod
    def init_artists(self):
        """Initialize artists for animation start."""

    @abstractmethod
    def render(self, frame_index: int):
        """Render one animation frame."""


class GPSDropoutAnimationBackend(AnimationBackend):
    """2x2 animation dashboard for the GPS dropout experiment."""

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def setup(self) -> None:
        t = self.data["t"]
        self.figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        self.axes = axes

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        dropout_start = self.data["dropout_start"]
        dropout_end = self.data["dropout_end"]

        # Position X
        ax = axes[0, 0]
        ax.set_xlim(t[0], t[-1])
        x_min = min(float(np.min(self.data["positions"][:, 0])), float(np.min(self.data["est_positions"][:, 0])))
        x_max = max(float(np.max(self.data["positions"][:, 0])), float(np.max(self.data["est_positions"][:, 0])))
        ax.set_ylim(x_min - 1.0, x_max + 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")
        ax.set_title("Position X")
        ax.axvspan(dropout_start, dropout_end, alpha=0.2, color="gray", label="GPS dropout")
        (self.truth_line,) = ax.plot([], [], "b-", label="Truth")
        (self.est_line,) = ax.plot([], [], "r--", label="Estimate", alpha=0.8)
        (self.cursor_x,) = ax.plot([t[0], t[0]], [x_min - 1.0, x_max + 1.0], "k:", alpha=0.5)
        ax.legend(loc="upper left")

        # Position error
        ax = axes[0, 1]
        ax.set_xlim(t[0], t[-1])
        err_max = float(np.max(self.data["pos_error"]))
        ax.set_ylim(0.0, max(1.0, err_max * 1.1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title("Position Error")
        ax.axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
        (self.err_line,) = ax.plot([], [], "g-", label="Error")
        (self.cursor_err,) = ax.plot([t[0], t[0]], [0.0, max(1.0, err_max * 1.1)], "k:", alpha=0.5)

        # Covariance trace
        ax = axes[1, 0]
        ax.set_xlim(t[0], t[-1])
        trace_max = float(np.max(self.data["p_trace"]))
        ax.set_ylim(0.0, max(1.0, trace_max * 1.1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("trace(P)")
        ax.set_title("Covariance Growth")
        ax.axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
        (self.trace_line,) = ax.plot([], [], "b-", label="trace(P)")
        (self.cursor_cov,) = ax.plot([t[0], t[0]], [0.0, max(1.0, trace_max * 1.1)], "k:", alpha=0.5)

        # NIS
        ax = axes[1, 1]
        ax.set_xlim(t[0], t[-1])
        nis_max = float(np.max(self.data["nis_values"])) if len(self.data["nis_values"]) else 5.0
        ax.set_ylim(0.0, max(6.0, nis_max * 1.2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIS")
        ax.set_title("NIS at GPS Updates")
        ax.axvspan(dropout_start, dropout_end, alpha=0.2, color="gray")
        ax.axhline(y=3.0, color="r", linestyle="--", label="dof=3 mean")
        (self.nis_line,) = ax.plot([], [], "o-", markersize=3, label="NIS")
        (self.cursor_nis,) = ax.plot([t[0], t[0]], [0.0, max(6.0, nis_max * 1.2)], "k:", alpha=0.5)
        ax.legend(loc="upper left")

        recovery_time = self.data.get("first_recovery_time")
        if recovery_time is not None:
            for ax in axes.flat:
                ax.axvline(recovery_time, color="orange", linestyle="--", alpha=0.6)

        self.figure.suptitle("GPS Dropout Simulation Playback")
        self.figure.tight_layout()

    def init_artists(self):
        self.truth_line.set_data([], [])
        self.est_line.set_data([], [])
        self.err_line.set_data([], [])
        self.trace_line.set_data([], [])
        self.nis_line.set_data([], [])
        return []

    def render(self, frame_index: int):
        t = self.data["t"]
        k = int(frame_index)
        t_now = t[k]

        self.truth_line.set_data(t[: k + 1], self.data["positions"][: k + 1, 0])
        self.est_line.set_data(t[: k + 1], self.data["est_positions"][: k + 1, 0])
        self.err_line.set_data(t[: k + 1], self.data["pos_error"][: k + 1])
        self.trace_line.set_data(t[: k + 1], self.data["p_trace"][: k + 1])

        nis_times = self.data["nis_times"]
        nis_values = self.data["nis_values"]
        if len(nis_times):
            valid = nis_times <= t_now
            self.nis_line.set_data(nis_times[valid], nis_values[valid])

        for cursor in (self.cursor_x, self.cursor_err, self.cursor_cov, self.cursor_nis):
            y0, y1 = cursor.axes.get_ylim()
            cursor.set_data([t_now, t_now], [y0, y1])

        return []


class UrbanCanyonAnimationBackend(AnimationBackend):
    """2x2 animation dashboard for urban canyon strategy comparison."""

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def setup(self) -> None:
        t = self.data["t"]
        self.figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        self.axes = axes

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        # Position X comparison
        ax = axes[0, 0]
        ax.set_xlim(t[0], t[-1])
        x_vals = [self.data["positions"][:, 0], self.data["est1"][:, 0], self.data["est2"][:, 0], self.data["est3"][:, 0]]
        x_min = min(float(np.min(x)) for x in x_vals)
        x_max = max(float(np.max(x)) for x in x_vals)
        ax.set_ylim(x_min - 1.0, x_max + 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")
        ax.set_title("Position X")
        (self.truth_line,) = ax.plot([], [], "b-", label="Truth")
        (self.est1_line,) = ax.plot([], [], "r--", alpha=0.8, label="No gating")
        (self.est2_line,) = ax.plot([], [], "g-.", alpha=0.8, label="Gating")
        (self.est3_line,) = ax.plot([], [], "m:", alpha=0.8, label="R x10")
        ax.legend(loc="upper left")

        # Error comparison
        ax = axes[0, 1]
        ax.set_xlim(t[0], t[-1])
        err_max = float(np.max(np.vstack([self.data["err1"], self.data["err2"], self.data["err3"]])))
        ax.set_ylim(0.0, max(1.0, err_max * 1.1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title("Position Error Norm")
        (self.err1_line,) = ax.plot([], [], "r-", label="No gating")
        (self.err2_line,) = ax.plot([], [], "g-", label="Gating")
        (self.err3_line,) = ax.plot([], [], "m-", label="R x10")
        ax.legend(loc="upper left")

        # NIS no gating
        ax = axes[1, 0]
        ax.set_xlim(t[0], t[-1])
        nis1_max = float(np.max(self.data["nis1_values"])) if len(self.data["nis1_values"]) else 8.0
        ax.set_ylim(0.0, max(8.0, nis1_max * 1.2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIS")
        ax.set_title("NIS (No gating)")
        ax.axhline(y=7.81, color="r", linestyle="--", label="chi2(0.95,3)")
        (self.nis1_line,) = ax.plot([], [], "o-", markersize=3, color="tab:blue")
        ax.legend(loc="upper left")

        # NIS with gating
        ax = axes[1, 1]
        ax.set_xlim(t[0], t[-1])
        nis2_max = float(np.max(self.data["nis2_values"])) if len(self.data["nis2_values"]) else 8.0
        ax.set_ylim(0.0, max(8.0, nis2_max * 1.2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIS")
        ax.set_title("NIS (With gating)")
        ax.axhline(y=7.81, color="r", linestyle="--", label="chi2(0.95,3)")
        (self.nis2_line,) = ax.plot([], [], "o-", markersize=3, color="tab:green")
        ax.legend(loc="upper left")

        self.figure.suptitle("Urban Canyon Strategy Playback")
        self.figure.tight_layout()

    def init_artists(self):
        self.truth_line.set_data([], [])
        self.est1_line.set_data([], [])
        self.est2_line.set_data([], [])
        self.est3_line.set_data([], [])
        self.err1_line.set_data([], [])
        self.err2_line.set_data([], [])
        self.err3_line.set_data([], [])
        self.nis1_line.set_data([], [])
        self.nis2_line.set_data([], [])
        return []

    def render(self, frame_index: int):
        t = self.data["t"]
        k = int(frame_index)
        t_now = t[k]

        self.truth_line.set_data(t[: k + 1], self.data["positions"][: k + 1, 0])
        self.est1_line.set_data(t[: k + 1], self.data["est1"][: k + 1, 0])
        self.est2_line.set_data(t[: k + 1], self.data["est2"][: k + 1, 0])
        self.est3_line.set_data(t[: k + 1], self.data["est3"][: k + 1, 0])

        self.err1_line.set_data(t[: k + 1], self.data["err1"][: k + 1])
        self.err2_line.set_data(t[: k + 1], self.data["err2"][: k + 1])
        self.err3_line.set_data(t[: k + 1], self.data["err3"][: k + 1])

        nis1_times = self.data["nis1_times"]
        nis1_values = self.data["nis1_values"]
        if len(nis1_times):
            v1 = nis1_times <= t_now
            self.nis1_line.set_data(nis1_times[v1], nis1_values[v1])

        nis2_times = self.data["nis2_times"]
        nis2_values = self.data["nis2_values"]
        if len(nis2_times):
            v2 = nis2_times <= t_now
            self.nis2_line.set_data(nis2_times[v2], nis2_values[v2])

        return []


class AdaptiveQAnimationBackend(AnimationBackend):
    """2x2 animation dashboard for adaptive-Q vs fixed-Q comparison."""

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def setup(self) -> None:
        t = self.data["t"]
        self.figure, axes = plt.subplots(2, 2, figsize=(12, 9))
        self.axes = axes

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        # Position X comparison
        ax = axes[0, 0]
        ax.set_xlim(t[0], t[-1])
        x_vals = [self.data["positions"][:, 0], self.data["est_fixed"][:, 0], self.data["est_adapt"][:, 0]]
        x_min = min(float(np.min(x)) for x in x_vals)
        x_max = max(float(np.max(x)) for x in x_vals)
        ax.set_ylim(x_min - 1.0, x_max + 1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("x (m)")
        ax.set_title("Position X")
        (self.truth_line,) = ax.plot([], [], "b-", label="Truth")
        (self.fixed_line,) = ax.plot([], [], "r--", alpha=0.8, label="Fixed Q")
        (self.adapt_line,) = ax.plot([], [], "g-.", alpha=0.8, label="Adaptive Q")
        ax.legend(loc="upper left")

        # Error comparison
        ax = axes[0, 1]
        ax.set_xlim(t[0], t[-1])
        err_max = float(np.max(np.vstack([self.data["err_fixed"], self.data["err_adapt"]])))
        ax.set_ylim(0.0, max(1.0, err_max * 1.1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.set_title("Position Error")
        (self.err_fixed_line,) = ax.plot([], [], "r-", label="Fixed Q")
        (self.err_adapt_line,) = ax.plot([], [], "g-", label="Adaptive Q")
        ax.legend(loc="upper left")

        # NIS fixed
        ax = axes[1, 0]
        ax.set_xlim(t[0], t[-1])
        nis_max = float(np.max(self.data["nis_fixed_values"])) if len(self.data["nis_fixed_values"]) else 8.0
        ax.set_ylim(0.0, max(8.0, nis_max * 1.2))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NIS")
        ax.set_title("NIS (Fixed Q)")
        ax.axhline(y=7.81, color="r", linestyle="--", label="chi2(0.95,3)")
        (self.nis_line,) = ax.plot([], [], "o-", markersize=3, color="tab:blue")
        ax.legend(loc="upper left")

        # Adaptive Q scale
        ax = axes[1, 1]
        ax.set_xlim(t[0], t[-1])
        q_max = float(np.max(self.data["q_scale"])) if len(self.data["q_scale"]) else 2.0
        ax.set_ylim(0.8, max(2.0, q_max * 1.1))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Q scale factor")
        ax.set_title("Adaptive Q Scale")
        (self.q_line,) = ax.plot([], [], "g-")

        self.figure.suptitle("Adaptive Q vs Fixed Q Playback")
        self.figure.tight_layout()

    def init_artists(self):
        self.truth_line.set_data([], [])
        self.fixed_line.set_data([], [])
        self.adapt_line.set_data([], [])
        self.err_fixed_line.set_data([], [])
        self.err_adapt_line.set_data([], [])
        self.nis_line.set_data([], [])
        self.q_line.set_data([], [])
        return []

    def render(self, frame_index: int):
        t = self.data["t"]
        k = int(frame_index)
        t_now = t[k]

        self.truth_line.set_data(t[: k + 1], self.data["positions"][: k + 1, 0])
        self.fixed_line.set_data(t[: k + 1], self.data["est_fixed"][: k + 1, 0])
        self.adapt_line.set_data(t[: k + 1], self.data["est_adapt"][: k + 1, 0])

        self.err_fixed_line.set_data(t[: k + 1], self.data["err_fixed"][: k + 1])
        self.err_adapt_line.set_data(t[: k + 1], self.data["err_adapt"][: k + 1])
        self.q_line.set_data(t[: k + 1], self.data["q_scale"][: k + 1])

        nis_times = self.data["nis_fixed_times"]
        nis_values = self.data["nis_fixed_values"]
        if len(nis_times):
            valid = nis_times <= t_now
            self.nis_line.set_data(nis_times[valid], nis_values[valid])

        return []
