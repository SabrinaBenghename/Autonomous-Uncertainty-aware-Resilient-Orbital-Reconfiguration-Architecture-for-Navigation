import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.orbit_dynamics import OrbitDynamics, R_EARTH
from src.sensors.gnss import GNSSSensor
from src.sensors.imu import IMUSensor
from src.sensors.star_tracker import StarTrackerSensor
from src.navigation.state_estimation import SpacecraftStateEstimator
from src.faults.fault_scenarios import BenchmarkScenarios
from src.ai.anomaly_detector import TelemetryAnomalyDetector
from src.ai.reconfiguration import AutonomousReconfigurationManager


def run_fault_scenario_comparison(scenario_id: int, duration: float = 2000.0, dt: float = 1.0) -> dict:
    scenario = BenchmarkScenarios.get_scenario(scenario_id)
    print(f"\n--- Running Experiment 02 (Scenario {scenario_id}: {scenario.name}) ---")

    r_orbit = R_EARTH + 550_000.0
    v_orbit = OrbitDynamics.get_circular_velocity(550_000.0)
    initial_state = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

    dynamics = OrbitDynamics(use_j2=True)
    t_eval = np.arange(0, duration, dt)
    true_states = dynamics.propagate(initial_state, (0, duration), t_eval)

    # 1. Standard EKF Run
    gnss_std = GNSSSensor(position_std=5.0, velocity_std=0.05, seed=42)
    imu_std = IMUSensor(seed=42)
    st_std = StarTrackerSensor(seed=42)
    injector_std = BenchmarkScenarios.setup_injector(scenario_id)
    est_std = SpacecraftStateEstimator(initial_state=initial_state)

    err_std = []
    for idx, t in enumerate(t_eval):
        est_std.propagate(dt)
        true_st = true_states[:, idx]
        fault_status = injector_std.apply_faults(t, gnss_sensor=gnss_std, imu_sensor=imu_std, star_tracker_sensor=st_std)

        meas = gnss_std.measure(true_st)
        est_std.update_gnss(meas)
        err = np.linalg.norm(est_std.get_estimated_state()[:3] - true_st[:3])
        err_std.append(err)

    rmse_std = np.sqrt(np.mean(np.array(err_std) ** 2))

    # 2. AI-Assisted Reconfiguration EKF Run
    gnss_ai = GNSSSensor(position_std=5.0, velocity_std=0.05, seed=42)
    imu_ai = IMUSensor(seed=42)
    st_ai = StarTrackerSensor(seed=42)
    injector_ai = BenchmarkScenarios.setup_injector(scenario_id)
    est_ai = SpacecraftStateEstimator(initial_state=initial_state)
    detector = TelemetryAnomalyDetector()
    reconfig_mgr = AutonomousReconfigurationManager()

    err_ai = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for idx, t in enumerate(t_eval):
        est_ai.propagate(dt)
        true_st = true_states[:, idx]
        fault_status = injector_ai.apply_faults(t, gnss_sensor=gnss_ai, imu_sensor=imu_ai, star_tracker_sensor=st_ai)

        meas = gnss_ai.measure(true_st)
        if meas["available"]:
            # Evaluate prior innovation to detect anomalies
            inno = est_ai.ekf.last_innovation
            cov = est_ai.ekf.last_innovation_cov
            anomaly_info = detector.detect_anomaly(inno, cov)

            # Determine ground truth fault status
            is_ground_truth_fault = fault_status["gnss_spoofed"] or fault_status["gnss_denied"]

            if anomaly_info["is_anomaly"] and is_ground_truth_fault:
                tp += 1
            elif anomaly_info["is_anomaly"] and not is_ground_truth_fault:
                fp += 1
            elif not anomaly_info["is_anomaly"] and is_ground_truth_fault:
                fn += 1
            else:
                tn += 1

            adj_cov_pos, adj_cov_vel, _ = reconfig_mgr.evaluate_and_reconfigure(
                t, anomaly_info, meas["cov_pos"], meas["cov_vel"]
            )
            est_ai.update_gnss(meas, R_pos_override=adj_cov_pos, R_vel_override=adj_cov_vel)

        err = np.linalg.norm(est_ai.get_estimated_state()[:3] - true_st[:3])
        err_ai.append(err)

    rmse_ai = np.sqrt(np.mean(np.array(err_ai) ** 2))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0

    print(f"Scenario {scenario_id} ({scenario.name}) Results:")
    print(f"  Standard EKF RMSE: {rmse_std:.2f} m")
    print(f"  AI-Assisted EKF RMSE: {rmse_ai:.2f} m")
    print(f"  Detection F1-Score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})")

    # Plot & Save
    os.makedirs("results/figures", exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_eval, err_std, 'r--', label=f"Standard EKF (RMSE={rmse_std:.1f}m)")
    plt.plot(t_eval, err_ai, 'b-', label=f"AI-Assisted EKF (RMSE={rmse_ai:.1f}m)")
    plt.ylabel("Position Error [m]")
    plt.xlabel("Time [s]")
    plt.title(f"AURORA Scenario {scenario_id}: {scenario.name}")
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join("results", "figures", f"exp_02_scenario_{scenario_id}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return {
        "scenario_id": scenario_id,
        "name": scenario.name,
        "rmse_std": rmse_std,
        "rmse_ai": rmse_ai,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    for s_id in range(2, 7):
        run_fault_scenario_comparison(s_id)
