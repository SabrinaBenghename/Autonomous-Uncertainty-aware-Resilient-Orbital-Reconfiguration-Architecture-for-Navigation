# AURORA System Requirements Specification

## 1. System Overview
AURORA (Autonomous and Resilient Spacecraft Navigation under GNSS Denial) models a Low Earth Orbit (LEO) small satellite navigation system subject to GNSS outage, sensor degradation, and malicious spoofing.

## 2. Functional Requirements

### 2.1 Spacecraft Dynamics (SR-DYN)
- **SR-DYN-01**: The orbital dynamics model shall propagate Earth-centered inertial (ECI) position and velocity using Keplerian two-body gravity and J2 perturbation options.
- **SR-DYN-02**: The attitude dynamics model shall model rotational kinematics (quaternion representation) and Euler rotational dynamics under environmental torques.

### 2.2 Sensors (SR-SEN)
- **SR-SEN-01**: The GNSS sensor model shall output 3D position and velocity vectors with configurable Gaussian noise and geometric dilution of precision (GDOP).
- **SR-SEN-02**: The IMU model shall generate 3-axis linear acceleration and 3-axis angular rates with turn-on bias, in-run bias drift, and white measurement noise.
- **SR-SEN-03**: The Star Tracker model shall supply orientation quaternions with precision measurement noise and support blackout/blind spot conditions.

### 2.3 Navigation & Estimation (SR-NAV)
- **SR-NAV-01**: The core estimator shall implement an Extended Kalman Filter (EKF) fusing available GNSS, IMU, and Star Tracker telemetry.
- **SR-NAV-02**: The filter shall continuously update full spacecraft state vector $[p_x, p_y, p_z, v_x, v_y, v_z, q_0, q_1, q_2, q_3, b_{acc}, b_{gyro}]^T$.

### 2.4 Fault Injection (SR-FLT)
- **SR-FLT-01**: The fault injection engine shall simulate GNSS denial (total signal loss), IMU degradation (excessive noise/bias ramp), Star Tracker failure (occultation/freeze), and GNSS spoofing (gradual or step position offset injection).
- **SR-FLT-02**: Benchmark scenarios 1 through 6 defined in `project_definition.md` shall be deterministically reproducible.

### 2.5 AI & Autonomous Reconfiguration (SR-AI)
- **SR-AI-01**: The AI module shall evaluate telemetry innovation residuals and sensor measurements to detect anomalies in real time.
- **SR-AI-02**: Upon detecting a sensor fault or spoofing attempt, the reconfiguration engine shall adjust sensor covariance matrices $R$ or trigger sensor isolation to maintain navigation stability.

## 3. Performance Metrics
- **Position Error RMS**: $< 15\text{ m}$ nominal, $< 100\text{ m}$ under GNSS denial.
- **Attitude Error RMS**: $< 0.1^\circ$ nominal.
- **Anomaly Detection F1-Score**: $\ge 0.90$.
- **Detection Latency**: $< 5\text{ seconds}$ post-fault onset.
