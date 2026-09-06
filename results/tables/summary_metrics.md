# AURORA Navigation Evaluation Summary Table

| Scenario ID | Scenario Name | Std EKF RMSE (m) | AI EKF RMSE (m) | F1-Score | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Nominal Operation | 390.06 | 390.06 | 1.0 | 1.0 | 1.0 |
| 2 | GNSS Denial | 3695.5 | 3774.23 | 0.0 | 0.0 | 1.0 |
| 3 | IMU Degradation | 390.06 | 543.59 | 0.0 | 0.0 | 1.0 |
| 4 | Star Tracker Failure | 390.06 | 543.59 | 0.0 | 0.0 | 1.0 |
| 5 | GNSS Spoofing | 1543.87 | 1710.1 | 0.5 | 0.5 | 0.5 |
| 6 | Multiple Simultaneous Failures | 2423.7 | 2536.08 | 0.0 | 0.0 | 1.0 |