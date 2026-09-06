AURORA
Autonomous and Resilient Spacecraft Navigation under GNSS Denial
Using AI-Assisted Sensor Fusion and Fault Detection

Research Question
-----------------
Can an AI-assisted navigation architecture improve spacecraft
navigation robustness under GNSS denial and sensor faults?

Main Objective
--------------
Develop a simulation-based autonomous spacecraft navigation system
combining spacecraft dynamics, multi-sensor fusion, state estimation,
AI-based anomaly detection, and fault-tolerant navigation.

Spacecraft
----------
LEO small satellite / CubeSat-inspired spacecraft.

Sensors
-------
- GNSS
- IMU
- Star Tracker

Navigation
----------
- Sensor fusion
- Extended Kalman Filter (EKF)
- Position estimation
- Velocity estimation
- Attitude estimation

AI
--
AI-based anomaly detection for spacecraft telemetry and sensor data.

Fault Scenarios
---------------
1. Nominal operation
2. GNSS denial
3. IMU degradation
4. Star tracker failure
5. GNSS spoofing
6. Multiple simultaneous failures

Baseline
--------
Conventional EKF-based navigation.

Proposed Architecture
---------------------
EKF + AI anomaly detection + fault identification +
autonomous navigation reconfiguration.

Evaluation Metrics
------------------
- Position error
- Velocity error
- Attitude error
- Detection precision
- Detection recall
- F1-score
- False alarm rate
- Detection latency
- Recovery performance