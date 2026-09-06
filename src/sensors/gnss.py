import numpy as np


class GNSSSensor:
    """
    Simulates a Spacecraft GNSS Receiver providing ECI Position and Velocity.
    Supports noise, outages (GNSS denial), and spoofing offsets.
    """

    def __init__(
        self,
        position_std: float = 5.0,     # meters
        velocity_std: float = 0.05,    # m/s
        seed: int = None
    ):
        self.pos_std = position_std
        self.vel_std = velocity_std
        self.rng = np.random.default_rng(seed)
        self.is_available = True
        self.spoof_offset = np.zeros(6)  # [dx, dy, dz, dvx, dvy, dvz]

    def set_availability(self, available: bool):
        """Enable or disable GNSS signal (denial mode)."""
        self.is_available = available

    def set_spoof_offset(self, offset: np.ndarray):
        """Inject position/velocity spoofing offset."""
        self.spoof_offset = np.array(offset, dtype=float)

    def measure(self, true_state: np.ndarray) -> dict:
        """
        Takes true state [px, py, pz, vx, vy, vz].
        Returns measurement dict containing position, velocity, availability, and noise covariance.
        """
        if not self.is_available:
            return {
                "available": False,
                "position": None,
                "velocity": None,
                "cov_pos": (self.pos_std ** 2) * np.eye(3),
                "cov_vel": (self.vel_std ** 2) * np.eye(3)
            }

        pos_noise = self.rng.normal(0, self.pos_std, 3)
        vel_noise = self.rng.normal(0, self.vel_std, 3)

        measured_pos = true_state[:3] + pos_noise + self.spoof_offset[:3]
        measured_vel = true_state[3:6] + vel_noise + self.spoof_offset[3:]

        return {
            "available": True,
            "position": measured_pos,
            "velocity": measured_vel,
            "cov_pos": (self.pos_std ** 2) * np.eye(3),
            "cov_vel": (self.vel_std ** 2) * np.eye(3)
        }
