import numpy as np


class IMUBiasFault:
    """
    Models IMU accelerometer & gyroscope bias step jumps, noise degradation, and drift faults.
    """

    def __init__(self, degradation_windows: list = None):
        self.degradation_windows = degradation_windows or []  # List of (t_start, t_end, factor)

    def add_degradation_window(self, t_start: float, t_end: float, factor: float = 10.0):
        """Add an IMU noise degradation window."""
        self.degradation_windows.append((t_start, t_end, factor))

    def evaluate(self, t: float, imu_sensor) -> dict:
        """
        Applies IMU noise degradation or bias step changes to imu_sensor at timestamp t.
        """
        is_degraded = False
        factor_applied = 1.0

        for t_start, t_end, factor in self.degradation_windows:
            if t_start <= t <= t_end:
                imu_sensor.set_degradation_factor(factor)
                is_degraded = True
                factor_applied = factor
                break

        if not is_degraded:
            imu_sensor.set_degradation_factor(1.0)

        return {"imu_degraded": is_degraded, "factor": factor_applied}
