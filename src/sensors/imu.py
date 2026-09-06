import numpy as np


class IMUSensor:
    """
    Simulates a 6-DOF Inertial Measurement Unit (Accelerometer + Gyroscope).
    Includes white measurement noise and dynamic bias random walk drift.
    """

    def __init__(
        self,
        accel_std: float = 1e-3,       # m/s^2
        gyro_std: float = 1e-4,        # rad/s
        accel_bias_std: float = 1e-4,  # m/s^2 / sqrt(s)
        gyro_bias_std: float = 1e-5,   # rad/s / sqrt(s)
        seed: int = None
    ):
        self.accel_std = accel_std
        self.gyro_std = gyro_std
        self.accel_bias_std = accel_bias_std
        self.gyro_bias_std = gyro_bias_std

        self.rng = np.random.default_rng(seed)

        self.accel_bias = self.rng.normal(0, 1e-3, 3)
        self.gyro_bias = self.rng.normal(0, 1e-4, 3)

        self.noise_multiplier = 1.0  # Used for IMU degradation scenarios

    def set_degradation_factor(self, factor: float):
        """Scale noise levels to simulate sensor degradation or thermal noise rise."""
        self.noise_multiplier = factor

    def step_bias(self, dt: float):
        """Update in-run bias random walk."""
        self.accel_bias += self.rng.normal(0, self.accel_bias_std * np.sqrt(dt), 3)
        self.gyro_bias += self.rng.normal(0, self.gyro_bias_std * np.sqrt(dt), 3)

    def measure(self, true_accel: np.ndarray, true_gyro: np.ndarray, dt: float) -> dict:
        """
        Takes true acceleration [m/s^2] and angular rates [rad/s].
        Returns measured values corrupted by bias and noise.
        """
        self.step_bias(dt)

        a_noise = self.rng.normal(0, self.accel_std * self.noise_multiplier, 3)
        w_noise = self.rng.normal(0, self.gyro_std * self.noise_multiplier, 3)

        meas_accel = true_accel + self.accel_bias + a_noise
        meas_gyro = true_gyro + self.gyro_bias + w_noise

        return {
            "accel": meas_accel,
            "gyro": meas_gyro,
            "accel_bias": self.accel_bias.copy(),
            "gyro_bias": self.gyro_bias.copy(),
            "cov_accel": ((self.accel_std * self.noise_multiplier) ** 2) * np.eye(3),
            "cov_gyro": ((self.gyro_std * self.noise_multiplier) ** 2) * np.eye(3)
        }
