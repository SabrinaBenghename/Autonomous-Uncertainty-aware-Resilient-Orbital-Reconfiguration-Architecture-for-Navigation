import numpy as np

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    propagate_orbit_with_j2_and_drag_like
)

import src.validation.full_mission_resilience as full_mission


# ============================================================
# AURORA
# Phase 17-B
#
# Full resilience under representative orbital geometries
#
#
# IMPORTANT DESIGN CHOICE
#
# We DO NOT modify the validated Phase-16 mission runner.
#
# Instead, this module reuses its validated subsystem
# components:
#
#   attitude MEKF
#   accelerometer simulation
#   GNSS + transmit-time FDIR
#   resilience manager
#   9-state navigation EKF
#
# while replacing only the orbital initial geometry.
#
#
# This preserves the Phase-16 architecture and allows
# orbital geometry to become the controlled variable.
# ============================================================


# ------------------------------------------------------------
# 1. REFERENCE SYSTEM CONDITION
#
# Controlled nominal engineering condition.
#
# Same values are used for every orbital geometry.
# Random measurement realizations change only through
# scenario_seed.
# ------------------------------------------------------------

def build_reference_system_scenario(
    scenario_seed
):

    return {
        "scenario_seed":
            int(
                scenario_seed
            ),

        # ====================================================
        # GNSS
        # ====================================================

        "pseudorange_noise_std_m":
            3.0,

        # ====================================================
        # ACCELEROMETER
        # ====================================================

        "accelerometer_noise_std_mps2":
            5.0e-7,

        "accelerometer_bias_x_mps2":
            1.0e-6,

        "accelerometer_bias_y_mps2":
            -0.8e-6,

        "accelerometer_bias_z_mps2":
            0.6e-6,

        # ====================================================
        # GYROSCOPE
        # ====================================================

        "gyro_noise_std_degps":
            5.0e-4,

        "gyro_bias_x_degps":
            1.0e-3,

        "gyro_bias_y_degps":
            -0.8e-3,

        "gyro_bias_z_degps":
            0.6e-3,

        # ====================================================
        # STAR TRACKER
        # ====================================================

        "star_tracker_noise_std_deg":
            0.02,

        # ====================================================
        # INITIAL NAVIGATION ERROR
        # ====================================================

        "initial_position_error_x_m":
            100.0,

        "initial_position_error_y_m":
            -80.0,

        "initial_position_error_z_m":
            60.0,

        "initial_velocity_error_x_mps":
            0.10,

        "initial_velocity_error_y_mps":
            -0.08,

        "initial_velocity_error_z_mps":
            0.06,

        # ====================================================
        # NON-GRAVITATIONAL FORCE
        # ====================================================

        "non_gravitational_acceleration_mps2":
            2.0e-5,

        # ====================================================
        # LOW-REDUNDANCY FAULT
        # ====================================================

        "fault_magnitude_m":
            50.0,

        "fault_start_min":
            105.0,

        "fault_duration_min":
            10.0,

        # ====================================================
        # OUTAGES
        # ====================================================

        "gnss_outage_duration_min":
            15.0,

        "star_tracker_outage_duration_min":
            10.0,

        "dual_outage_duration_min":
            15.0
    }


# ------------------------------------------------------------
# 2. CUSTOM ORBIT TRUTH
# ------------------------------------------------------------

def build_orbit_specific_truth(
    orbit_scenario,
    base_acceleration
):

    orbit_scenario = dict(
        orbit_scenario
    )


    altitude_m = (
        float(
            orbit_scenario[
                "altitude_km"
            ]
        )
        *
        1000.0
    )


    semi_major_axis = (
        R_EARTH
        +
        altitude_m
    )


    eccentricity = float(
        orbit_scenario.get(
            "eccentricity",
            0.01
        )
    )


    inclination = np.deg2rad(
        float(
            orbit_scenario[
                "inclination_deg"
            ]
        )
    )


    raan = np.deg2rad(
        float(
            orbit_scenario.get(
                "raan_deg",
                40.0
            )
        )
    )


    argument_of_periapsis = np.deg2rad(
        float(
            orbit_scenario.get(
                "argument_of_periapsis_deg",
                30.0
            )
        )
    )


    true_anomaly = np.deg2rad(
        float(
            orbit_scenario[
                "initial_true_anomaly_deg"
            ]
        )
    )


    (
        initial_position,
        initial_velocity
    ) = keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )


    initial_state = np.concatenate(
        (
            initial_position,
            initial_velocity
        )
    )


    duration_seconds = (
        full_mission.MISSION_DURATION_MIN
        *
        60.0
    )


    number_of_epochs = (
        int(
            duration_seconds
            /
            full_mission.DT
        )
        +
        1
    )


    time = np.linspace(
        0.0,
        duration_seconds,
        number_of_epochs
    )


    solution = (
        propagate_orbit_with_j2_and_drag_like(
            initial_state=
                initial_state,

            duration=
                duration_seconds,

            number_of_points=
                number_of_epochs,

            base_acceleration=
                float(
                    base_acceleration
                )
        )
    )


    truth_states = (
        solution.y.T
    )


    synchronization_error = float(
        np.max(
            np.abs(
                solution.t
                -
                time
            )
        )
    )


    return {
        "time":
            time,

        "time_minutes":
            time
            /
            60.0,

        "truth_states":
            truth_states,

        "initial_orbital_state":
            initial_state,

        "synchronization_error":
            synchronization_error
    }


# ------------------------------------------------------------
# 3. SAFE PERCENTAGE
# ------------------------------------------------------------

def _percentage(
    values
):

    values = np.asarray(
        values,
        dtype=bool
    )


    if values.size == 0:

        return np.nan


    return float(
        100.0
        *
        np.mean(
            values
        )
    )


# ------------------------------------------------------------
# 4. RUN ONE ORBIT / NOISE REALIZATION
# ------------------------------------------------------------

def run_orbit_resilience_scenario(
    orbit_scenario,
    replicate_index,
    scenario_seed
):

    orbit_scenario = dict(
        orbit_scenario
    )


    system_scenario = (
        build_reference_system_scenario(
            scenario_seed=
                scenario_seed
        )
    )


    # ========================================================
    # CUSTOM ORBIT
    # ========================================================

    truth = (
        build_orbit_specific_truth(
            orbit_scenario=
                orbit_scenario,

            base_acceleration=
                system_scenario[
                    "non_gravitational_acceleration_mps2"
                ]
        )
    )


    # ========================================================
    # REUSE VALIDATED PHASE-16 PIPELINE
    # ========================================================

    true_attitude = (
        full_mission._build_true_attitude(
            truth[
                "truth_states"
            ]
        )
    )


    event_masks = (
        full_mission._build_event_masks(
            truth[
                "time_minutes"
            ],

            system_scenario
        )
    )


    attitude_result = (
        full_mission._simulate_attitude(
            scenario=
                system_scenario,

            event_masks=
                event_masks,

            true_attitude=
                true_attitude
        )
    )


    accelerometer_measurements = (
        full_mission._simulate_accelerometer(
            scenario=
                system_scenario,

            truth=
                truth,

            true_attitude=
                true_attitude
        )
    )


    initial_navigation_position = (
        truth[
            "initial_orbital_state"
        ][
            0:3
        ]
        +
        np.array(
            [
                system_scenario[
                    "initial_position_error_x_m"
                ],

                system_scenario[
                    "initial_position_error_y_m"
                ],

                system_scenario[
                    "initial_position_error_z_m"
                ]
            ],
            dtype=float
        )
    )


    gnss_result = (
        full_mission._simulate_gnss_fdir(
            scenario=
                system_scenario,

            truth=
                truth,

            event_masks=
                event_masks,

            initial_navigation_position=
                initial_navigation_position
        )
    )


    permissive = (
        full_mission._run_navigation_branch(
            scenario=
                system_scenario,

            truth=
                truth,

            event_masks=
                event_masks,

            attitude_result=
                attitude_result,

            accelerometer_measurements=
                accelerometer_measurements,

            gnss_result=
                gnss_result,

            protected=
                False
        )
    )


    protected = (
        full_mission._run_navigation_branch(
            scenario=
                system_scenario,

            truth=
                truth,

            event_masks=
                event_masks,

            attitude_result=
                attitude_result,

            accelerometer_measurements=
                accelerometer_measurements,

            gnss_result=
                gnss_result,

            protected=
                True
        )
    )


    permissive_metrics = (
        full_mission._policy_metrics(
            permissive,
            event_masks
        )
    )


    protected_metrics = (
        full_mission._policy_metrics(
            protected,
            event_masks
        )
    )


    # ========================================================
    # GNSS GEOMETRY / FDIR METRICS
    # ========================================================

    signal_mask = (
        gnss_result[
            "signal_available"
        ]
    )


    strong_mask = (
        event_masks[
            "strong_fault"
        ]
        &
        signal_mask
    )


    low_mask = (
        event_masks[
            "low_redundancy_fault"
        ]
        &
        signal_mask
    )


    healthy_mask = (
        signal_mask
        &
        ~gnss_result[
            "intentional_fault"
        ]
    )


    if np.any(
        strong_mask
    ):

        strong_detection_rate = (
            _percentage(
                gnss_result[
                    "fault_detected"
                ][
                    strong_mask
                ]
            )
        )


        strong_isolation_rate = (
            _percentage(
                gnss_result[
                    "correct_isolation"
                ][
                    strong_mask
                ]
            )
        )

    else:

        strong_detection_rate = (
            np.nan
        )


        strong_isolation_rate = (
            np.nan
        )


    if np.any(
        low_mask
    ):

        low_detection_rate = (
            _percentage(
                gnss_result[
                    "fault_detected"
                ][
                    low_mask
                ]
            )
        )

    else:

        low_detection_rate = (
            np.nan
        )


    if np.any(
        healthy_mask
    ):

        false_alarm_rate = (
            _percentage(
                gnss_result[
                    "fault_detected"
                ][
                    healthy_mask
                ]
            )
        )

    else:

        false_alarm_rate = (
            np.nan
        )


    # ========================================================
    # NATURAL SATELLITE COUNT
    #
    # Exclude the intentionally forced five-satellite
    # low-redundancy window.
    # ========================================================

    natural_geometry_mask = (
        signal_mask
        &
        ~event_masks[
            "low_redundancy_fault"
        ]
    )


    natural_satellite_counts = (
        gnss_result[
            "satellite_counts"
        ][
            natural_geometry_mask
        ]
    )


    if len(
        natural_satellite_counts
    ) > 0:

        natural_visible_mean = float(
            np.mean(
                natural_satellite_counts
            )
        )


        natural_visible_min = int(
            np.min(
                natural_satellite_counts
            )
        )

    else:

        natural_visible_mean = (
            np.nan
        )


        natural_visible_min = (
            -1
        )


    # ========================================================
    # ATTITUDE
    # ========================================================

    attitude_errors_deg = np.rad2deg(
        attitude_result[
            "attitude_errors"
        ]
    )


    attitude_rmse_deg = float(
        np.sqrt(
            np.mean(
                attitude_errors_deg**2
            )
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "geometry_name":
            str(
                orbit_scenario[
                    "geometry_name"
                ]
            ),

        "source_scenario_id":
            str(
                orbit_scenario.get(
                    "source_scenario_id",
                    "MANUAL"
                )
            ),

        "replicate_index":
            int(
                replicate_index
            ),

        "scenario_seed":
            int(
                scenario_seed
            ),

        "altitude_km":
            float(
                orbit_scenario[
                    "altitude_km"
                ]
            ),

        "inclination_deg":
            float(
                orbit_scenario[
                    "inclination_deg"
                ]
            ),

        "initial_true_anomaly_deg":
            float(
                orbit_scenario[
                    "initial_true_anomaly_deg"
                ]
            ),

        "time_sync_error_s":
            float(
                truth[
                    "synchronization_error"
                ]
            ),

        # ----------------------------------------------------
        # NATURAL GNSS GEOMETRY
        # ----------------------------------------------------

        "natural_visible_satellites_min":
            natural_visible_min,

        "natural_visible_satellites_mean":
            natural_visible_mean,

        # ----------------------------------------------------
        # FDIR
        # ----------------------------------------------------

        "strong_detection_rate":
            strong_detection_rate,

        "strong_isolation_rate":
            strong_isolation_rate,

        "low_detection_rate":
            low_detection_rate,

        "false_alarm_rate":
            false_alarm_rate,

        # ----------------------------------------------------
        # ATTITUDE
        # ----------------------------------------------------

        "attitude_rmse_deg":
            attitude_rmse_deg,

        # ----------------------------------------------------
        # PERMISSIVE
        # ----------------------------------------------------

        **{
            f"permissive_{key}":
                value

            for key, value in
            permissive_metrics.items()
        },

        # ----------------------------------------------------
        # PROTECTED
        # ----------------------------------------------------

        **{
            f"protected_{key}":
                value

            for key, value in
            protected_metrics.items()
        },

        # ----------------------------------------------------
        # PAIRED POLICY BENEFIT
        # ----------------------------------------------------

        "gain_mission_rmse_m":
            (
                permissive_metrics[
                    "mission_rmse"
                ]
                -
                protected_metrics[
                    "mission_rmse"
                ]
            ),

        "gain_low_rmse_m":
            (
                permissive_metrics[
                    "low_redundancy_rmse"
                ]
                -
                protected_metrics[
                    "low_redundancy_rmse"
                ]
            ),

        "gain_dual_rmse_m":
            (
                permissive_metrics[
                    "dual_outage_rmse"
                ]
                -
                protected_metrics[
                    "dual_outage_rmse"
                ]
            )
    }


    return result