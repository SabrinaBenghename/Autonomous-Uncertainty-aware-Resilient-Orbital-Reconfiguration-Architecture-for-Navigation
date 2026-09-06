import numpy as np


class GNSSPointSolver:
    """
    GNSS Point Positioning Solver.
    Computes spacecraft ECI position and receiver clock bias from pseudorange measurements
    or solves 3D position state updates given raw GNSS observations.
    """

    def __init__(self, speed_of_light: float = 299792458.0):
        self.c = speed_of_light

    def solve_least_squares(
        self,
        sat_positions: np.ndarray,
        pseudoranges: np.ndarray,
        initial_guess: np.ndarray = None,
        max_iter: int = 10,
        tol: float = 1e-4
    ) -> tuple:
        """
        Solves 3D user position [x, y, z] and receiver clock offset dt_rx [meters]
        given satellite positions (N x 3) and pseudoranges (N x 1).
        """
        N = len(sat_positions)
        if N < 4:
            raise ValueError("At least 4 satellites required for 3D GNSS solver.")

        if initial_guess is None:
            x = np.zeros(4)
        else:
            x = np.append(initial_guess[:3], 0.0)

        for _ in range(max_iter):
            pos = x[:3]
            dt_rx = x[3]

            rho_calc = np.linalg.norm(sat_positions - pos, axis=1) + dt_rx
            residual = pseudoranges - rho_calc

            # Geometry Matrix H (N x 4)
            H = np.zeros((N, 4))
            for i in range(N):
                r_mag = np.linalg.norm(sat_positions[i] - pos)
                H[i, :3] = -(sat_positions[i] - pos) / r_mag
                H[i, 3] = 1.0

            # Solve Least Squares delta_x = (H^T H)^-1 H^T residual
            delta_x, _, _, _ = np.linalg.lstsq(H, residual, rcond=None)
            x += delta_x

            if np.linalg.norm(delta_x) < tol:
                break

        return x[:3], x[3] / self.c

    def filter_raw_position(self, raw_position: np.ndarray, noise_std: float = 5.0) -> dict:
        """
        Provides GNSS solver output metadata for point positioning.
        """
        return {
            "position": raw_position,
            "covariance": (noise_std ** 2) * np.eye(3),
            "status": "SOLVED"
        }
