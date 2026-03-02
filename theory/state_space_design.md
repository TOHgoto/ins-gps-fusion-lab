# INS-GPS State Space Design

## Overview

This document defines the state vector, process model, and measurement model for the INS-GPS fusion system. The model is linear in the error state, suitable for a linear Kalman filter.

---

## State Vector

The state vector has 12 components:

```
x = [p_x, p_y, p_z, v_x, v_y, v_z, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]^T
```

| Index | Symbol | Physical meaning |
|-------|--------|------------------|
| 0-2   | p      | Position (m) in world frame |
| 3-5   | v      | Velocity (m/s) |
| 6-8   | b_a    | Accelerometer bias (m/s^2) |
| 9-11  | b_g    | Gyroscope bias (rad/s) |

---

## Continuous-Time Process Model

We use a simplified linear model in the inertial frame (no rotation dynamics for the basic case):

```
p_dot = v
v_dot = a - b_a + w_a        (a = true acceleration, w_a = accel white noise)
b_a_dot = w_ba               (bias random walk)
b_g_dot = w_bg               (bias random walk)
```

The IMU measures `a_meas = a + b_a + eta_a`, so the filter estimates `b_a` to correct the acceleration. For a linear position-velocity model, we treat `a` as the known input (from IMU) and `b_a` as an estimated bias to subtract.

**Simplified for linear KF**: We model the *error state* or use a constant-velocity model with acceleration as process noise. For a minimal implementation:

```
p_dot = v
v_dot = a_meas - b_a         (a_meas from IMU, b_a estimated)
b_a_dot = 0 + w_ba          (random walk driven by w_ba)
b_g_dot = 0 + w_bg          (random walk driven by w_bg)
```

Process noise: `w = [w_a, w_ba, w_bg]` with covariances derived from IMU specs.

---

## Discrete-Time Model

### Transition Matrix F

Using Euler discretization with time step dt:

```
p[k+1] = p[k] + v[k] * dt
v[k+1] = v[k] + (a_meas - b_a) * dt  =>  v[k+1] = v[k] - b_a * dt  (treating a_meas as input)
b_a[k+1] = b_a[k] + w_ba
b_g[k+1] = b_g[k] + w_bg
```

For the *state propagation* (without process noise), the transition is:

- p: p + v*dt
- v: v - b_a*dt  (when a_meas=0 in error model) or v + a*dt - b_a*dt
- b_a: b_a
- b_g: b_g

F matrix (12x12):

```
F = [ I3   dt*I3   0    0   ]
    [ 0    I3     -dt*I3  0   ]
    [ 0    0      I3     0   ]
    [ 0    0      0     I3   ]
```

Where I3 is 3x3 identity.

### Process Noise Input Matrix G

Process noise enters as:
- w_a (3): affects velocity
- w_ba (3): affects b_a
- w_bg (3): affects b_g

```
G = [ 0     0    0 ]
    [ dt*I3 0    0 ]
    [ 0     I3   0 ]
    [ 0     0    I3 ]
```

### Process Noise Covariance Q

Q = G * diag(Q_a, Q_ba, Q_bg) * G^T

Where:
- Q_a: covariance of velocity random walk from accel (sigma_acc^2 * dt * I3)
- Q_ba: covariance of accel bias random walk (sigma_acc_rw^2 * dt * I3)
- Q_bg: covariance of gyro bias random walk (sigma_gyro_rw^2 * dt * I3)

---

## Measurement Model

GPS provides position only:

```
z = H * x + v,    H = [ I3  0  0  0 ]
```

H is 3x12: we observe the first 3 components (position).

---

## Observability

- Position is directly observed by GPS.
- Velocity is observable through the coupling p_dot = v (position rate gives velocity).
- Accelerometer bias b_a affects velocity evolution; with persistent excitation (non-zero acceleration), b_a is observable.
- Gyro bias b_g does not appear in position/velocity dynamics in this simplified linear model; it would matter in a full attitude model. For position-velocity fusion, b_g may be unobservable.

For the minimal p-v-b_a model with GPS position, the system is observable when there is sufficient motion.
