import numpy as np
from src.navigation.ekf import ExtendedKalmanFilter


class SpacecraftStateEstimator:
    """
    High-level spacecraft state estimation manager.
    Coordinates measurement updates from GNSS, IMU, and Star Tracker.
    """

    def __init__(self, initial_state: np.ndarray):
        self.ekf = ExtendedKalmanFilter(initial_state=initial_state)
        self.estimated_state_history = []
        self.covariance_history = []

    def propagate(self, dt: float):
        """Propagate state forward by dt."""
        self.ekf.predict(dt)
        self.estimated_state_history.append(self.ekf.x.flatten().copy())
        self.covariance_history.append(np.diag(self.ekf.P).copy())

    def update_gnss(self, gnss_data: dict, R_pos_override: np.ndarray = None, R_vel_override: np.ndarray = None):
        """
        Process GNSS measurement if available.
        gnss_data contains 'position', 'velocity', 'available', 'cov_pos', 'cov_vel'.
        """
        if not gnss_data.get("available", False):
            return None, None

        pos = gnss_data["position"]
        vel = gnss_data["velocity"]

        z = np.concatenate([pos, vel])
        H = np.eye(6)

        R_pos = R_pos_override if R_pos_override is not None else gnss_data["cov_pos"]
        R_vel = R_vel_override if R_vel_override is not None else gnss_data["cov_vel"]

        R = np.block([
            [R_pos, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_vel]
        ])

        return self.ekf.update(z=z, H=H, R=R)

    def get_estimated_state(self) -> np.ndarray:
        """Returns current state vector estimate [px, py, pz, vx, vy, vz]."""
        return self.ekf.x.flatten()

    def get_estimated_covariance(self) -> np.ndarray:
        """Returns current 6x6 state covariance matrix."""
        return self.ekf.P.copy()
