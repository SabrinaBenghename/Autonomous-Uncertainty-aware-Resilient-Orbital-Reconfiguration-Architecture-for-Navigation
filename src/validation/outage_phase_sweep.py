import numpy as np

import src.validation.full_mission_resilience as full_mission

from src.validation.orbit_resilience_sweep import (
    build_reference_system_scenario,
    build_orbit_specific_truth
)


# ============================================================
# AURORA
# Phase 17-C
#
# Dual-outage orbital-phase sweep
#
#
# Controlled variables:
#
#       altitude      = 550 km
#       inclination   = 97.6 deg
#       eccentricity  = 0.01
#       RAAN          = 40 deg
#       arg periapsis = 30 deg
#
#
# Variable:
#
#       initial true anomaly
#
#
# Fixed dual outage:
#
#       60 - 75 min
#
#
# Because outage time is fixed while initial orbital phase
# changes, the spacecraft encounters the same outage after
# exactly the same estimator-learning duration but at
# different locations/phases along its orbit.
#
#
# No GNSS fault or other artificial outage is present.
#
# This isolates:
#
#       orbital phase -> autonomous outage performance
# ============================================================


# ------------------------------------------------------------
# 1. SWEEP
# ------------------------------------------------------------

PHASES_DEG = [
    0.0,
    45.0,
    90.0,
    135.0,
    180.0,
    225.0,
    270.0,
    315.0
]


DUAL_OUTAGE_START_MIN = (
    60.0
)


DUAL_OUTAGE_DURATION_MIN = (
    15.0
)


DUAL_OUTAGE_END_MIN = (
    DUAL_OUTAGE_START_MIN
    +
    DUAL_OUTAGE_DURATION_MIN
)


POST_RECOVERY_END_MIN = (
    90.0
)


# ------------------------------------------------------------
# 2. RMSE
# ------------------------------------------------------------

def _rmse(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )


    if values.size == 0:

        return np.nan


    return float(
        np.sqrt(
            np.mean(
                values**2
            )
        )
    )


# ------------------------------------------------------------
# 3. EVENT MASKS
#
# Only the dual GNSS + star-tracker outage is active.
# ------------------------------------------------------------

def build_phase_sweep_event_masks(
    time_minutes
):

    time_minutes = np.asarray(
        time_minutes,
        dtype=float
    )


    number_of_epochs = len(
        time_minutes
    )


    false_mask = np.zeros(
        number_of_epochs,
        dtype=bool
    )


    dual_outage = (
        (
            time_minutes
            >=
            DUAL_OUTAGE_START_MIN
        )
        &
        (
            time_minutes
            <
            DUAL_OUTAGE_END_MIN
        )
    )


    post_dual_outage = (
        (
            time_minutes
            >=
            DUAL_OUTAGE_END_MIN
        )
        &
        (
            time_minutes
            <
            POST_RECOVERY_END_MIN
        )
    )


    burn_in = (
        time_minutes
        >=
        full_mission.BURN_IN_MIN
    )


    return {
        "strong_fault":
            false_mask.copy(),

        "early_gnss_outage":
            false_mask.copy(),

        "star_tracker_outage":
            false_mask.copy(),

        "low_redundancy_fault":
            false_mask.copy(),

        "post_low_redundancy":
            false_mask.copy(),

        "dual_outage":
            dual_outage,

        "post_dual_outage":
            post_dual_outage,

        "burn_in":
            burn_in,

        "low_fault_start":
            np.nan,

        "low_fault_end":
            np.nan,

        "dual_outage_end":
            DUAL_OUTAGE_END_MIN
    }


# ------------------------------------------------------------
# 4. ORBIT
# ------------------------------------------------------------

def build_reference_orbit_with_phase(
    initial_true_anomaly_deg
):

    return {
        "geometry_name":
            (
                f"PHASE_"
                f"{int(round(initial_true_anomaly_deg)):03d}"
            ),

        "source_scenario_id":
            "017C_PHASE_SWEEP",

        "altitude_km":
            550.0,

        "inclination_deg":
            97.6,

        "initial_true_anomaly_deg":
            float(
                initial_true_anomaly_deg
            ),

        "eccentricity":
            0.01,

        "raan_deg":
            40.0,

        "argument_of_periapsis_deg":
            30.0
    }


# ------------------------------------------------------------
# 5. LAST FINITE VALUE BEFORE INDEX
# ------------------------------------------------------------

def _last_finite_before(
    values,
    index
):

    values = np.asarray(
        values,
        dtype=float
    )


    if index <= 0:

        return np.nan


    candidate_indices = np.where(
        np.isfinite(
            values[
                :index
            ]
        )
    )[0]


    if len(
        candidate_indices
    ) == 0:

        return np.nan


    return float(
        values[
            candidate_indices[
                -1
            ]
        ]
    )


# ------------------------------------------------------------
# 6. LAST UPDATE AGE
# ------------------------------------------------------------

def _last_update_age_seconds(
    update_used,
    outage_start_index,
    dt
):

    update_used = np.asarray(
        update_used,
        dtype=bool
    )


    previous_updates = np.where(
        update_used[
            :outage_start_index
        ]
    )[0]


    if len(
        previous_updates
    ) == 0:

        return np.nan


    last_update_index = (
        previous_updates[
            -1
        ]
    )


    return float(
        (
            outage_start_index
            -
            last_update_index
        )
        *
        dt
    )


# ------------------------------------------------------------
# 7. ONE PHASE / ONE STOCHASTIC REPLICATE
# ------------------------------------------------------------

def run_dual_outage_phase_scenario(
    initial_true_anomaly_deg,
    replicate_index,
    scenario_seed,
    estimator_start_min=0.0
):

    # ========================================================
    # Same system condition as 017-B / 017-C.
    # ========================================================

    system_scenario = (
        build_reference_system_scenario(
            scenario_seed=
                scenario_seed
        )
    )


    orbit_scenario = (
        build_reference_orbit_with_phase(
            initial_true_anomaly_deg=
                initial_true_anomaly_deg
        )
    )


    # ========================================================
    # Truth
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


    time_minutes = (
        truth[
            "time_minutes"
        ]
    )


    event_masks = (
        build_phase_sweep_event_masks(
            time_minutes
        )
    )


    # ========================================================
    # True attitude
    #
    # IMPORTANT:
    #
    # The attitude MEKF remains active for the whole mission.
    #
    # 017-F specifically manipulates NAVIGATION-EKF maturity,
    # because navigation covariance was the mechanism
    # identified by 017-D / 017-E.
    # ========================================================

    true_attitude = (
        full_mission._build_true_attitude(
            truth[
                "truth_states"
            ]
        )
    )


    # ========================================================
    # MEKF
    # ========================================================

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


    # ========================================================
    # Body-frame accelerometer
    # ========================================================

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


    # ========================================================
    # Initial GNSS navigation prior
    # ========================================================

    initial_position_error = np.array(
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


    initial_navigation_position = (
        truth[
            "initial_orbital_state"
        ][
            0:3
        ]
        +
        initial_position_error
    )


    # ========================================================
    # Healthy GNSS outside dual outage
    #
    # GNSS measurement generation is intentionally unchanged
    # across estimator-maturity conditions.
    # ========================================================

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


    # ========================================================
    # AURORA protected navigation
    #
    # NEW:
    #
    # estimator_start_min controls when the navigation EKF
    # actually begins.
    # ========================================================

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
                True,

            estimator_start_min=
                float(
                    estimator_start_min
                )
        )
    )


    # ========================================================
    # Masks
    # ========================================================

    dual_mask = (
        event_masks[
            "dual_outage"
        ]
    )


    post_mask = (
        event_masks[
            "post_dual_outage"
        ]
    )


    burn_in_mask = (
        event_masks[
            "burn_in"
        ]
    )


    pre_outage_mask = (
        (
            time_minutes
            >=
            45.0
        )
        &
        (
            time_minutes
            <
            DUAL_OUTAGE_START_MIN
        )
    )


    dual_indices = np.where(
        dual_mask
    )[0]


    if len(
        dual_indices
    ) == 0:

        raise RuntimeError(
            "Dual-outage mask is empty."
        )


    outage_start_index = int(
        dual_indices[
            0
        ]
    )


    outage_end_index = int(
        dual_indices[
            -1
        ]
    )


    # ========================================================
    # NAVIGATION ESTIMATOR CONTROL VALIDATION
    # ========================================================

    actual_estimator_start_min = float(
        protected[
            "estimator_start_time_min"
        ]
    )


    actual_estimator_age_min = float(
        protected[
            "estimator_age_at_dual_outage_min"
        ]
    )


    requested_estimator_start_min = float(
        estimator_start_min
    )


    # With DT = 10 s, requested times used by 017-F should
    # land exactly on simulation epochs.
    max_start_timing_error_min = (
        full_mission.DT
        /
        60.0
        +
        1.0e-12
    )


    if (
        abs(
            actual_estimator_start_min
            -
            requested_estimator_start_min
        )
        >
        max_start_timing_error_min
    ):

        raise RuntimeError(
            "Navigation estimator activation timing "
            "does not match requested 017-F control."
        )


    # ========================================================
    # Navigation errors
    # ========================================================

    position_errors = (
        protected[
            "position_errors"
        ]
    )


    position_sigmas = (
        protected[
            "position_sigmas"
        ]
    )


    # ========================================================
    # Attitude
    # ========================================================

    attitude_error_deg = np.rad2deg(
        attitude_result[
            "attitude_errors"
        ]
    )


    # ========================================================
    # Healthy-GNSS false alarms
    # ========================================================

    healthy_signal_mask = (
        gnss_result[
            "signal_available"
        ]
    )


    if np.any(
        healthy_signal_mask
    ):

        false_alarm_rate = float(
            100.0
            *
            np.mean(
                gnss_result[
                    "fault_detected"
                ][
                    healthy_signal_mask
                ]
            )
        )


    else:

        false_alarm_rate = np.nan


    # ========================================================
    # Natural geometry outside outage
    # ========================================================

    natural_counts = (
        gnss_result[
            "satellite_counts"
        ][
            healthy_signal_mask
        ]
    )


    if len(
        natural_counts
    ) > 0:

        natural_visible_mean = float(
            np.mean(
                natural_counts
            )
        )


        natural_visible_min = int(
            np.min(
                natural_counts
            )
        )


    else:

        natural_visible_mean = np.nan

        natural_visible_min = -1


    # ========================================================
    # Geometry immediately before outage
    # ========================================================

    pre_outage_pdop = (
        _last_finite_before(
            gnss_result[
                "pdop"
            ],

            outage_start_index
        )
    )


    if outage_start_index > 0:

        pre_outage_satellites = int(
            gnss_result[
                "satellite_counts"
            ][
                outage_start_index
                -
                1
            ]
        )


    else:

        pre_outage_satellites = -1


    update_age_seconds = (
        _last_update_age_seconds(
            update_used=
                protected[
                    "update_used"
                ],

            outage_start_index=
                outage_start_index,

            dt=
                full_mission.DT
        )
    )


    # ========================================================
    # Count estimator-active epochs before outage
    # ========================================================

    estimator_active_pre_outage_epochs = int(
        np.sum(
            protected[
                "estimator_active"
            ][
                :outage_start_index
            ]
        )
    )


    estimator_updates_before_outage = int(
        np.sum(
            protected[
                "update_used"
            ][
                :outage_start_index
            ]
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "phase_name":
            orbit_scenario[
                "geometry_name"
            ],

        "initial_true_anomaly_deg":
            float(
                initial_true_anomaly_deg
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
            550.0,

        "inclination_deg":
            97.6,

        # ----------------------------------------------------
        # OUTAGE DESIGN
        # ----------------------------------------------------

        "dual_outage_start_min":
            DUAL_OUTAGE_START_MIN,

        "dual_outage_end_min":
            DUAL_OUTAGE_END_MIN,

        "dual_outage_duration_min":
            DUAL_OUTAGE_DURATION_MIN,

        # ----------------------------------------------------
        # NAVIGATION-ESTIMATOR INTERVENTION
        # ----------------------------------------------------

        "navigation_estimator_requested_start_min":
            requested_estimator_start_min,

        "navigation_estimator_start_min":
            actual_estimator_start_min,

        "navigation_estimator_start_index":
            int(
                protected[
                    "estimator_start_index"
                ]
            ),

        "navigation_estimator_age_at_outage_min":
            actual_estimator_age_min,

        "navigation_estimator_active_pre_outage_epochs":
            estimator_active_pre_outage_epochs,

        "navigation_estimator_updates_before_outage":
            estimator_updates_before_outage,

        # ----------------------------------------------------
        # SYNCHRONIZATION
        # ----------------------------------------------------

        "time_sync_error_s":
            float(
                truth[
                    "synchronization_error"
                ]
            ),

        # ----------------------------------------------------
        # NATURAL GEOMETRY
        # ----------------------------------------------------

        "natural_visible_satellites_min":
            natural_visible_min,

        "natural_visible_satellites_mean":
            natural_visible_mean,

        "pre_outage_satellites":
            pre_outage_satellites,

        "pre_outage_pdop":
            pre_outage_pdop,

        "pre_outage_update_age_s":
            update_age_seconds,

        # ----------------------------------------------------
        # GNSS HEALTH
        # ----------------------------------------------------

        "healthy_false_alarm_rate":
            false_alarm_rate,

        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        "pre_outage_rmse_m":
            _rmse(
                position_errors[
                    pre_outage_mask
                ]
            ),

        "mission_rmse_m":
            _rmse(
                position_errors[
                    burn_in_mask
                ]
            ),

        "outage_rmse_m":
            _rmse(
                position_errors[
                    dual_mask
                ]
            ),

        "outage_start_error_m":
            float(
                position_errors[
                    outage_start_index
                ]
            ),

        "outage_end_error_m":
            float(
                position_errors[
                    outage_end_index
                ]
            ),

        "outage_start_sigma_m":
            float(
                position_sigmas[
                    outage_start_index
                ]
            ),

        "outage_end_sigma_m":
            float(
                position_sigmas[
                    outage_end_index
                ]
            ),

        "post_recovery_rmse_m":
            _rmse(
                position_errors[
                    post_mask
                ]
            ),

        "final_error_m":
            float(
                position_errors[
                    -1
                ]
            ),

        "navigation_nis_mean":
            float(
                protected[
                    "nis_mean"
                ]
            ),

        "navigation_nis_coverage_percent":
            float(
                protected[
                    "nis_coverage"
                ]
            ),

        # ----------------------------------------------------
        # ATTITUDE
        # ----------------------------------------------------

        "attitude_outage_rmse_deg":
            _rmse(
                attitude_error_deg[
                    dual_mask
                ]
            ),

        "attitude_end_outage_deg":
            float(
                attitude_error_deg[
                    outage_end_index
                ]
            )
    }


    return result