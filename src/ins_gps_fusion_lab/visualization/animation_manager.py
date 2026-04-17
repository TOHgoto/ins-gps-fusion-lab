"""Animation orchestration utilities for simulation playback."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from ins_gps_fusion_lab.visualization.animation_backends import AnimationBackend


class AnimationManager:
    """Coordinate playback and export for simulation animations."""

    def __init__(self, backend: AnimationBackend, frame_indices: Iterable[int], fps: int = 20):
        self.backend = backend
        self.frame_indices = list(frame_indices)
        self.fps = fps
        self._anim: animation.FuncAnimation | None = None

    def _ensure_animation(self) -> animation.FuncAnimation:
        if self._anim is not None:
            return self._anim

        self.backend.setup()

        interval_ms = max(1, int(1000 / max(self.fps, 1)))

        self._anim = animation.FuncAnimation(
            self.backend.figure,
            self.backend.render,
            init_func=self.backend.init_artists,
            frames=self.frame_indices,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        return self._anim

    def show(self) -> None:
        """Play animation in an interactive window."""
        self._ensure_animation()
        plt.show()

    def save(self, output_path: str) -> None:
        """Save animation to .mp4 or .gif based on output suffix."""
        anim = self._ensure_animation()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix = target.suffix.lower()
        if suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=self.fps)
        elif suffix == ".gif":
            writer = animation.PillowWriter(fps=self.fps)
        else:
            raise ValueError("output_path must end with .mp4 or .gif")

        anim.save(str(target), writer=writer)
