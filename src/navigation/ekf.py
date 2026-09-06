import numpy as np
from src.dynamics.orbit_dynamics import MU_EARTH


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for spacecraft state estimation.
    State vector (6x1): [px, py, pz, vx, vy, vz]^T in ECI frame.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray = None,
        process_noise_std_pos: float = 1e-2,
        process_noise_std_vel: float = 1e-4
    ):
        self.x = np.array(initial_state, dtype=float).reshape(6, 1)

        if initial_covariance is None:
            self.P = np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        else:
            self.P = np.array(initial_covariance, dtype=float)

        self.q_pos = process_noise_std_pos
        self.q_vel = process_noise_std_vel

        # Last innovation residual vector and covariance for AI anomaly detection
        self.last_innovation = None
        self.last_innovation_cov = None

    def state_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Computes Jacobian matrix F (6x6) of Keplerian two-body dynamics.
        df/dx = [ 0   I ]
                [ G   0 ]
        where G_ij = d(a_i)/d(r_j) = -mu/r^3 (delta_ij - 3 r_i r_j / r^2)
        """
        r = state[:3].flatten()
        r_norm = np.linalg.norm(r)
        mu_r3 = MU_EARTH / (r_norm ** 3)
        mu_r5 = MU_EARTH / (r_norm ** 5)

        G = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                delta = 1.0 if i == j else 0.0
                G[i, j] = -mu_r3 * delta + 3.0 * mu_r5 * r[i] * r[j]

        F = np.zeros((6, 6))
        F[:3, 3:] = np.eye(3)
        F[3:, :3] = G
        return F

    def predict(self, dt: float):
        """
        Predict step over interval dt using 2-body orbital integration and state transition matrix F.
        """
        r = self.x[:3, 0]
        v = self.x[3:, 0]
        r_norm = np.linalg.norm(r)

        # Acceleration
        a = - (MU_EARTH / (r_norm ** 3)) * r

        # Euler step for state prediction
        self.x[:3, 0] += v * dt + 0.5 * a * (dt ** 2)
        self.x[3:, 0] += a * dt

        # Linearized State Transition Matrix Phi ~ I + F * dt
        F = self.state_jacobian(self.x)
        Phi = np.eye(6) + F * dt + 0.5 * (F @ F) * (dt ** 2)

        # Process noise matrix Q
        Q_pos = (self.q_pos ** 2) * np.eye(3) * dt
        Q_vel = (self.q_vel ** 2) * np.eye(3) * dt
        Q = np.block([
            [Q_pos, np.zeros((3, 3))],
            [np.zeros((3, 3)), Q_vel]
        ])

        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> tuple:
        """
        Update step with measurement vector z, measurement matrix H, and measurement noise covariance R.
        Returns (innovation, innovation_covariance).
        """
        z = np.array(z, dtype=float).reshape(-1, 1)

        # Predicted measurement
        z_pred = H @ self.x

        # Innovation (residual)
        y = z - z_pred
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.last_innovation = y
        self.last_innovation_cov = S

        return y, S
