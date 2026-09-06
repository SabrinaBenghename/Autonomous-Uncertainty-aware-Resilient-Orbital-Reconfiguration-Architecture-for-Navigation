import numpy as np
from src.ai.anomaly_detector import TelemetryAnomalyDetector
from src.ai.reconfiguration import AutonomousReconfigurationManager


def test_anomaly_detection_mahalanobis():
    detector = TelemetryAnomalyDetector(chi2_threshold=16.81)

    # Nominal innovation (near zero)
    inno_nominal = np.array([0.1, -0.2, 0.05, 0.0, 0.01, -0.01])
    cov = np.eye(6) * 1.0

    res_nom = detector.detect_anomaly(inno_nominal, cov)
    assert res_nom["is_anomaly"] is False

    # Large anomalous innovation (spoofing offset)
    inno_faulty = np.array([500.0, 500.0, 500.0, 0.0, 0.0, 0.0])
    res_faulty = detector.detect_anomaly(inno_faulty, cov)
    assert res_faulty["is_anomaly"] is True


def test_reconfiguration_manager():
    mgr = AutonomousReconfigurationManager(inflation_factor=1e6)
    base_cov_pos = np.eye(3) * 25.0
    base_cov_vel = np.eye(3) * 0.01

    anomaly_info = {"is_anomaly": True, "reason": "Chi2 threshold exceeded"}
    adj_pos, adj_vel, active = mgr.evaluate_and_reconfigure(10.0, anomaly_info, base_cov_pos, base_cov_vel)

    assert active is True
    assert np.all(adj_pos == base_cov_pos * 1e6)
    assert np.all(adj_vel == base_cov_vel * 1e6)
