import numpy as np
from src.sensors.gnss import GNSSSensor
from src.sensors.imu import IMUSensor
from src.sensors.star_tracker import StarTrackerSensor


def test_gnss_nominal_and_denial():
    gnss = GNSSSensor(position_std=5.0, velocity_std=0.05, seed=42)
    true_state = np.array([7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])

    meas = gnss.measure(true_state)
    assert meas["available"] is True
    assert np.linalg.norm(meas["position"] - true_state[:3]) < 25.0  # 5-sigma bound

    # Test Denial Mode
    gnss.set_availability(False)
    meas_denied = gnss.measure(true_state)
    assert meas_denied["available"] is False
    assert meas_denied["position"] is None


def test_gnss_spoofing():
    gnss = GNSSSensor(seed=42)
    true_state = np.array([7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])
    offset = np.array([500.0, 500.0, 500.0, 0.0, 0.0, 0.0])
    gnss.set_spoof_offset(offset)

    meas = gnss.measure(true_state)
    assert meas["available"] is True
    # Position should reflect the ~500m spoofing offset
    assert np.all(meas["position"] > true_state[:3] + 400.0)


def test_imu_measurement():
    imu = IMUSensor(seed=42)
    true_acc = np.array([0.0, 0.0, 9.81])
    true_gyro = np.array([0.001, 0.0, 0.0])

    meas = imu.measure(true_acc, true_gyro, dt=0.1)
    assert "accel" in meas
    assert "gyro" in meas
    assert meas["accel"].shape == (3,)
    assert meas["gyro"].shape == (3,)


def test_star_tracker_measurement():
    st = StarTrackerSensor(angle_std=1e-4, seed=42)
    q_true = np.array([1.0, 0.0, 0.0, 0.0])

    meas = st.measure(q_true)
    assert meas["available"] is True
    np.testing.assert_allclose(np.linalg.norm(meas["quaternion"]), 1.0, atol=1e-6)

    st.set_availability(False)
    meas_outage = st.measure(q_true)
    assert meas_outage["available"] is False
