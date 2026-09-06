import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure root workspace is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dynamics.orbit_dynamics import OrbitDynamics, R_EARTH
from src.sensors.gnss import GNSSSensor
from src.sensors.imu import IMUSensor
from src.navigation.state_estimation import SpacecraftStateEstimator


def run_nominal_experiment(duration: float = 2000.0, dt: float = 1.0) -> dict:
    print("--- Running Experiment 01: Nominal Operations ---")

    alt = 550_000.0
    r_orbit = R_EARTH + alt
    v_orbit = OrbitDynamics.get_circular_velocity(alt)

    # Initial state [px, py, pz, vx, vy, vz]
    initial_state = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

    dynamics = OrbitDynamics(use_j2=True)
    t_eval = np.arange(0, duration, dt)
    true_states = dynamics.propagate(initial_state, (0, duration), t_eval)

    gnss = GNSSSensor(position_std=5.0, velocity_std=0.05, seed=42)
    estimator = SpacecraftStateEstimator(initial_state=initial_state)

    est_positions = []
    pos_errors = []

    for idx, t in enumerate(t_eval):
        estimator.propagate(dt)
        true_st = true_states[:, idx]

        meas = gnss.measure(true_st)
        estimator.update_gnss(meas)

        est_st = estimator.get_estimated_state()
        est_positions.append(est_st[:3])

        pos_err = np.linalg.norm(est_st[:3] - true_st[:3])
        pos_errors.append(pos_err)

    pos_rmse = np.sqrt(np.mean(np.array(pos_errors) ** 2))
    print(f"Nominal Position RMSE: {pos_rmse:.3f} meters")

    # Plot & Save
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t_eval, true_states[0, :] / 1000.0, label="True X (km)")
    ax[0].plot(t_eval, np.array(est_positions)[:, 0] / 1000.0, '--', label="Est X (km)")
    ax[0].set_ylabel("ECI X [km]")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(t_eval, pos_errors, color='crimson', label="Position Error (m)")
    ax[1].set_ylabel("Error [m]")
    ax[1].set_xlabel("Time [s]")
    ax[1].legend()
    ax[1].grid(True)
    ax[0].set_title("AURORA Exp 01 - Nominal Spacecraft State Estimation")

    fig_path = os.path.join("results", "figures", "exp_01_nominal.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure to {fig_path}")

    return {"t_eval": t_eval, "pos_errors": pos_errors, "pos_rmse": pos_rmse}


if __name__ == "__main__":
    run_nominal_experiment()
