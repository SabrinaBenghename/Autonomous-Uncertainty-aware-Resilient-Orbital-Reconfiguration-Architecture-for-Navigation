import numpy as np
from src.navigation.ekf import ExtendedKalmanFilter
from src.navigation.state_estimation import SpacecraftStateEstimator
from src.navigation.gnss_solver import GNSSPointSolver


def test_ekf_predict_update():
    x0 = np.array([7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])
    ekf = ExtendedKalmanFilter(initial_state=x0)

    P_prior = np.trace(ekf.P)
    ekf.predict(dt=1.0)
    P_predicted = np.trace(ekf.P)

    # Uncertainty increases during prediction
    assert P_predicted > P_prior

    # Update with GNSS measurement
    z = x0 + np.random.normal(0, 1.0, 6)
    H = np.eye(6)
    R = np.eye(6) * 25.0

    ekf.update(z, H, R)
    P_updated = np.trace(ekf.P)

    # Uncertainty decreases after measurement update
    assert P_updated < P_predicted


def test_gnss_point_solver():
    solver = GNSSPointSolver()
    true_pos = np.array([7000000.0, 10000.0, 5000.0])

    # 4 dummy satellites
    sats = np.array([
        [7100000.0, 10000.0, 5000.0],
        [7000000.0, 11000.0, 5000.0],
        [7000000.0, 10000.0, 6000.0],
        [7050000.0, 10500.0, 5500.0]
    ])

    ranges = np.linalg.norm(sats - true_pos, axis=1)
    pos_est, dt_rx = solver.solve_least_squares(sats, ranges, initial_guess=true_pos + 10.0)

    np.testing.assert_allclose(pos_est, true_pos, atol=1e-2)


def test_spacecraft_state_estimator():
    x0 = np.array([7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])
    estimator = SpacecraftStateEstimator(initial_state=x0)

    estimator.propagate(1.0)
    est_state = estimator.get_estimated_state()
    assert est_state.shape == (6,)
