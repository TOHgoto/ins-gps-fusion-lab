# INS-GPS Fusion Lab

**INS-GPS Simulation & Fusion Suite** — an open-source, reusable sensor fusion research and validation toolkit.

## Positioning

- **Target users**: Researchers, engineers, students — for Kalman filter learning, algorithm validation, prototyping, and teaching
- **Coverage**: IMU error modeling, GPS modeling, state space design, Kalman derivation, covariance propagation, Q/R tuning, NIS test, Innovation Gating, GPS dropout, urban canyon simulation, adaptive Q
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

### Run examples

```bash
# Basic INS-GPS fusion (trajectory + IMU + GPS + Kalman filter)
python -m experiments.run_basic_fusion

# GPS dropout (20s loss at 20-40s)
python -m experiments.gps_dropout

# Q mismatch (filter inconsistency)
python -m experiments.test_q_mismatch

# Urban canyon (outliers, gating comparison)
python -m experiments.urban_canyon

# Adaptive Q vs fixed Q
python -m experiments.adaptive_q

# Run tests
pytest tests/

# Run tests and save results to results/test_reports/
python scripts/run_tests.py
```

Test results are stored in `results/test_reports/`:
- `summary.md` — human-readable summary by module
- `summary.json` — machine-readable JSON
- `junit.xml` — for CI integration
- `pytest_output.txt` — raw pytest output

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
├── docs/             # Mathematical derivations and design docs
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
