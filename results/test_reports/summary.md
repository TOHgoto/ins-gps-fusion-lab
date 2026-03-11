# Test Results Summary

**Run**: 2026-03-12 05:35 | **Status**: PASS | **Total**: 29 tests in 0.82s

---

## Overall

| Metric | Value |
|--------|-------|
| Passed | 29 |
| Failed | 0 |
| Skipped | 0 |
| Total time | 0.82 s |

---

## By Module

### 1. Simulation — GPS Model (`test_gps_model.py`)

| Test | Status |
|------|--------|
| test_gps_reproducibility | PASS |
| test_gps_output_shape | PASS |
| test_gps_batch_output_shape | PASS |
| test_gps_zero_noise_gives_truth | PASS |
| test_gps_dropout_produces_invalid | PASS |
| test_gps_outlier_increases_error | PASS |
| test_gps_compute_residual | PASS |
| test_gps_sigma_per_axis | PASS |

**8 passed, 0 failed**

---

### 2. Simulation — IMU Model (`test_imu_model.py`)

| Test | Status |
|------|--------|
| test_imu_reproducibility | PASS |
| test_imu_output_shape | PASS |
| test_imu_bias_grows_with_random_walk | PASS |
| test_imu_zero_noise_gives_truth | PASS |
| test_imu_reset_bias | PASS |
| test_imu_plot_error_growth_runs | PASS |

**6 passed, 0 failed**

---

### 3. Filters — Kalman Filter (`test_kalman_filter.py`)

| Test | Status |
|------|--------|
| test_predict_only_p_grows | PASS |
| test_update_shrinks_p | PASS |
| test_nis_chi_squared | PASS |
| test_reproducibility | PASS |
| test_update_with_correct_measurement | PASS |
| test_innovation_gating_rejects_outlier | PASS |

**6 passed, 0 failed**

---

### 4. Simulation — State Space (`test_state_space.py`)

| Test | Status |
|------|--------|
| test_F_shape | PASS |
| test_F_identity_blocks | PASS |
| test_F_p_v_coupling | PASS |
| test_H_position_only | PASS |
| test_Q_positive_semidefinite | PASS |
| test_G_shape | PASS |

**6 passed, 0 failed**

---

### 5. Simulation — Trajectory Generator (`test_trajectory_generator.py`)

| Test | Status |
|------|--------|
| test_straight_line_shape | PASS |
| test_straight_line_constant_velocity | PASS |
| test_straight_line_position_integration | PASS |

**3 passed, 0 failed**

---

## File Layout

```
results/
└── test_reports/
    ├── summary.md
    ├── summary.json
    ├── junit.xml
    └── pytest_output.txt
```