# INS-GPS Fusion Lab

**INS-GPS Simulation & Fusion Suite** — an open-source, reusable sensor fusion research and validation toolkit.

<p align="center">
    <a href="#key-results-animated-">
        <img alt="Key Results" src="https://img.shields.io/badge/Key%20Results-Animated%20Demo-0A7E8C?style=for-the-badge" />
    </a>
    <a href="#installation">
        <img alt="Installation" src="https://img.shields.io/badge/Setup-Installation-1F6FEB?style=for-the-badge" />
    </a>
    <a href="#usage">
        <img alt="Usage" src="https://img.shields.io/badge/Run-Usage-2E8B57?style=for-the-badge" />
    </a>
    <a href="#project-structure">
        <img alt="Project Structure" src="https://img.shields.io/badge/Code-Project%20Structure-6B7280?style=for-the-badge" />
    </a>
</p>

## Key Results (Animated) 🎬

Showcase GIFs are served from `media/readme/`.

### GPS Dropout Recovery 📉

<p>
    <img alt="Category" src="https://img.shields.io/badge/Category-Robustness-1F6FEB?style=flat-square" />
</p>

<p>
    IMU-only drift grows during GPS outage, then quickly contracts after measurement recovery.
</p>
<p align="center">
    <a href="media/readme/gps_dropout.gif">
        <img src="media/readme/gps_dropout.gif" width="88%" alt="GPS dropout and recovery card" />
    </a>
</p>

### Urban Canyon Robustness 🏙️

<p>
    <img alt="Category" src="https://img.shields.io/badge/Category-Outlier%20Handling-1F6FEB?style=flat-square" />
</p>

<p>
    Innovation gating suppresses outlier injections and keeps trajectory estimation stable.
</p>
<p align="center">
    <a href="media/readme/urban_canyon.gif">
        <img src="media/readme/urban_canyon.gif" width="88%" alt="Urban canyon robustness card" />
    </a>
</p>

### Adaptive Q Under Mismatch ⚙️

<p>
    <img alt="Category" src="https://img.shields.io/badge/Category-Adaptivity-1F6FEB?style=flat-square" />
</p>

<p>
    Dynamic process-noise scaling improves consistency when the nominal Q model is too optimistic.
</p>
<p align="center">
    <a href="media/readme/adaptive_q.gif">
        <img src="media/readme/adaptive_q.gif" width="88%" alt="Adaptive Q mismatch card" />
    </a>
</p>

## Positioning

- **Target users**: Researchers, engineers, students — for Kalman filter learning, algorithm validation, prototyping, and teaching
- **Coverage**: IMU and GPS modeling, state space design, Q/R tuning, NIS test, innovation gating, GPS dropout, urban canyon simulation, adaptive Q
- **Engineering value**: Reproducible experiments, configurable simulation, extensible filter interface, ready for algorithm comparison and tuning

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ins-gps-fusion-lab.git
cd ins-gps-fusion-lab

# Install in development mode (recommended)
pip install -e ".[dev]"
```

## Usage

### Quick run

```bash
# Fast sanity check
python -m experiments.run_basic_fusion

# Main reliability demo
python -m experiments.gps_dropout
```

<details>
<summary>More experiment commands</summary>

```bash
# Q mismatch (filter inconsistency)
python -m experiments.test_q_mismatch

# Urban canyon (outliers, gating comparison)
python -m experiments.urban_canyon

# Adaptive Q vs fixed Q
python -m experiments.adaptive_q

# Run tests
pytest tests/

# Optional: run tests and generate local reports under results/test_reports/
python scripts/run_tests.py
```

</details>

### Animated simulation playback

```python
from experiments.gps_dropout import run_animation as run_dropout_animation
from experiments.urban_canyon import run_animation as run_urban_animation
from experiments.adaptive_q import run_animation as run_adaptive_q_animation

# GPS dropout playback (save GIF + show window)
run_dropout_animation(
    duration=30.0,
    frame_stride=20,
    fps=20,
    output_path="results/figures/gps_dropout.gif",
    show=True,
)

# Urban canyon strategy playback (save GIF, no interactive window)
run_urban_animation(
    duration=20.0,
    frame_stride=10,
    fps=20,
    output_path="results/figures/urban_canyon.gif",
    show=False,
)

# Adaptive Q playback (save GIF)
run_adaptive_q_animation(
    duration=30.0,
    frame_stride=10,
    fps=20,
    output_path="results/figures/adaptive_q.gif",
    show=False,
)
```

### Use as a library

```python
from ins_gps_fusion_lab.simulation.imu_model import IMUModel
from ins_gps_fusion_lab.simulation.gps_model import GPSModel

# Create IMU model with configurable noise parameters
imu = IMUModel(sigma_acc=0.1, sigma_gyro=0.01, seed=42)
acc_meas, gyro_meas = imu.generate_measurements(truth_acc, truth_gyro, dt=0.01)

# Create GPS model with optional outlier/dropout
gps = GPSModel(sigma_pos=1.0, outlier_prob=0.01)
position_meas, valid = gps.generate_measurement(true_position)
```

## Project Structure

```
ins-gps-fusion-lab/
├── docs/             # Theory and design notes
├── src/
│   └── ins_gps_fusion_lab/
│       ├── simulation/       # IMU, GPS, trajectory models
│       ├── filters/          # Kalman filter, EKF
│       └── visualization/    # Plotting utilities
├── experiments/      # Runnable experiments
└── tests/            # Unit tests
```

## License

MIT License — see [LICENSE](LICENSE) for details.
