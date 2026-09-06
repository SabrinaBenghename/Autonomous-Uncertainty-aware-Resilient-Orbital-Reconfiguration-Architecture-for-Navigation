import numpy as np

from scipy.stats import chi2

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    synthetic_drag_like_acceleration,
    propagate_orbit_with_j2_and_drag_like
)

from src.attitude.reference_attitude import (
    build_nadir_pointing_body_to_eci
)

from src.attitude.quaternions import (
    rotation_matrix_to_quaternion,
    rotation_matrix_to_rotation_vector,
    quaternion_to_rotation_matrix,
    quaternion_attitude_error_angle
)

from src.attitude.attitude_mekf import (
    predict_attitude_mekf,
    update_attitude_mekf_with_star_tracker
)

from src.sensors.gyroscope import (
    simulate_gyroscope_measurement
)

from src.sensors.star_tracker import (
    simulate_star_tracker_measurement
)

from src.sensors.body_frame_accelerometer import (
    simulate_body_frame_accelerometer_measurement
)

from src.navigation.body_bias_ekf import (
    ekf_predict_with_body_bias_estimation
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_update_position_augmented
)

from src.navigation.gnss_ekf_interface import (
    generate_simplified_gps_positions,
    select_visible_gnss_satellites
)

from src.navigation.gnss_signal_propagation import (
    simulate_pseudoranges_with_transmit_time,
    compute_transmit_time_solution_covariance
)

from src.navigation.gnss_fdir_transmit_time import (
    solve_gnss_with_fdir_transmit_time
)

from src.faults.resilience_manager import (
    ResilienceManager
)


# ============================================================
# AURORA
# Phase 16-B
#
# Reusable full-mission robustness runner
#
#
# Architecture:
#
#   truth:
#       J2 + synthetic non-gravitational acceleration
#
#   attitude:
#       gyro + star tracker MEKF
#
#   translation:
#       body-frame accelerometer
#       9-state [r, v, b_a,B] EKF
#
#   GNSS:
#       transmit-time-corrected pseudoranges
#       LS
#       statistical FDIR
#
#   policies:
#
#       PERMISSIVE
#           current FDIR only
#
#       PROTECTED
#           FDIR + integrity availability
#
#
# This runner deliberately excludes AI.
#
# AI robustness is Phase 16-C.
# ============================================================


# ------------------------------------------------------------
# 1. FIXED MISSION CONSTANTS
# ------------------------------------------------------------

MISSION_DURATION_MIN = (
    150.0
)

DT = (
    10.0
)

BURN_IN_MIN = (
    10.0
)

MINIMUM_SATELLITES_FOR_INTEGRITY = (
    6
)

STRONG_FAULT_START_MIN = (
    20.0
)

STRONG_FAULT_END_MIN = (
    30.0
)

STRONG_FAULT_MAGNITUDE_M = (
    50.0
)

EARLY_GNSS_OUTAGE_START_MIN = (
    40.0
)

STAR_TRACKER_OUTAGE_START_MIN = (
    65.0
)

DUAL_OUTAGE_START_MIN = (
    125.0
)

PREFERRED_FAULT_SATELLITE = (
    "GPS04"
)

RECEIVER_CLOCK_BIAS_SECONDS = (
    100.0e-6
)

GNSS_CONFIDENCE = (
    0.99
)

STAR_TRACKER_PERIOD_SECONDS = (
    60.0
)

GYRO_BIAS_RANDOM_WALK_DENSITY = np.deg2rad(
    1.0e-7
)

ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY = (
    1.0e-10
)

RECOVERY_EPOCHS = (
    6
)

ATTITUDE_SIGMA_LIMIT_RAD = np.deg2rad(
    0.5
)


# ------------------------------------------------------------
# 2. REFERENCE ORBIT
#
# Phase 17 will vary orbital geometry.
# ------------------------------------------------------------

SEMI_MAJOR_AXIS = (
    R_EARTH
    +
    550_000.0
)

ECCENTRICITY = (
    0.01
)

INCLINATION = np.deg2rad(
    97.6
)

RAAN = np.deg2rad(
    40.0
)

ARGUMENT_OF_PERIAPSIS = np.deg2rad(
    30.0
)

TRUE_ANOMALY = np.deg2rad(
    25.0
)


# ------------------------------------------------------------
# 3. HELPERS
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


def _safe_mean(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    valid = np.isfinite(
        values
    )

    if not np.any(
        valid
    ):

        return np.nan

    return float(
        np.mean(
            values[
                valid
            ]
        )
    )


def _percentage(
    boolean_values
):

    boolean_values = np.asarray(
        boolean_values,
        dtype=bool
    )

    if boolean_values.size == 0:

        return np.nan

    return float(
        100.0
        *
        np.mean(
            boolean_values
        )
    )


def _select_five_satellites(
    visible_ids
):

    visible_ids = [
        str(
            satellite_id
        ).strip()
        for satellite_id in visible_ids
    ]

    if len(
        visible_ids
    ) < 5:

        return None

    if (
        PREFERRED_FAULT_SATELLITE
        in
        visible_ids
    ):

        others = [
            satellite_id
            for satellite_id in visible_ids
            if satellite_id
            !=
            PREFERRED_FAULT_SATELLITE
        ]

        return (
            [
                PREFERRED_FAULT_SATELLITE
            ]
            +
            others[
                :4
            ]
        )

    return (
        visible_ids[
            :5
        ]
    )


def _choose_fault_target(
    satellite_ids
):

    if (
        PREFERRED_FAULT_SATELLITE
        in
        satellite_ids
    ):

        return (
            PREFERRED_FAULT_SATELLITE
        )

    return (
        satellite_ids[
            0
        ]
    )


# ------------------------------------------------------------
# 4. BUILD TRUTH
# ------------------------------------------------------------

def _build_truth(
    base_acceleration
):

    duration_seconds = (
        MISSION_DURATION_MIN
        *
        60.0
    )

    number_of_epochs = (
        int(
            duration_seconds
            /
            DT
        )
        +
        1
    )

    time = np.linspace(
        0.0,
        duration_seconds,
        number_of_epochs
    )

    time_minutes = (
        time
        /
        60.0
    )

    (
        initial_position,
        initial_velocity
    ) = keplerian_to_cartesian(
        SEMI_MAJOR_AXIS,
        ECCENTRICITY,
        INCLINATION,
        RAAN,
        ARGUMENT_OF_PERIAPSIS,
        TRUE_ANOMALY
    )

    initial_orbital_state = np.concatenate(
        (
            initial_position,
            initial_velocity
        )
    )

    solution = (
        propagate_orbit_with_j2_and_drag_like(
            initial_state=
                initial_orbital_state,

            duration=
                duration_seconds,

            number_of_points=
                number_of_epochs,

            base_acceleration=
                base_acceleration
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
            time_minutes,

        "truth_states":
            truth_states,

        "initial_orbital_state":
            initial_orbital_state,

        "synchronization_error":
            synchronization_error
    }


# ------------------------------------------------------------
# 5. TRUE ATTITUDE
# ------------------------------------------------------------

def _build_true_attitude(
    truth_states
):

    number_of_epochs = (
        len(
            truth_states
        )
    )

    rotations = np.zeros(
        (
            number_of_epochs,
            3,
            3
        )
    )

    quaternions = np.zeros(
        (
            number_of_epochs,
            4
        )
    )

    for index in range(
        number_of_epochs
    ):

        rotations[
            index
        ] = (
            build_nadir_pointing_body_to_eci(
                position_eci=
                    truth_states[
                        index,
                        0:3
                    ],

                velocity_eci=
                    truth_states[
                        index,
                        3:6
                    ]
            )
        )

        quaternions[
            index
        ] = (
            rotation_matrix_to_quaternion(
                rotations[
                    index
                ]
            )
        )

    body_rates = np.zeros(
        (
            number_of_epochs - 1,
            3
        )
    )

    for index in range(
        number_of_epochs - 1
    ):

        relative_rotation = (
            rotations[
                index
            ].T
            @
            rotations[
                index + 1
            ]
        )

        rotation_vector = (
            rotation_matrix_to_rotation_vector(
                relative_rotation
            )
        )

        body_rates[
            index
        ] = (
            rotation_vector
            /
            DT
        )

    return {
        "rotations":
            rotations,

        "quaternions":
            quaternions,

        "body_rates":
            body_rates
    }


# ------------------------------------------------------------
# 6. EVENT MASKS
# ------------------------------------------------------------

def _build_event_masks(
    time_minutes,
    scenario
):

    early_gnss_outage_end = (
        EARLY_GNSS_OUTAGE_START_MIN
        +
        float(
            scenario[
                "gnss_outage_duration_min"
            ]
        )
    )

    star_tracker_outage_end = (
        STAR_TRACKER_OUTAGE_START_MIN
        +
        float(
            scenario[
                "star_tracker_outage_duration_min"
            ]
        )
    )

    low_fault_start = float(
        scenario[
            "fault_start_min"
        ]
    )

    low_fault_end = (
        low_fault_start
        +
        float(
            scenario[
                "fault_duration_min"
            ]
        )
    )

    dual_outage_end = (
        DUAL_OUTAGE_START_MIN
        +
        float(
            scenario[
                "dual_outage_duration_min"
            ]
        )
    )

    strong_fault = (
        (
            time_minutes
            >=
            STRONG_FAULT_START_MIN
        )
        &
        (
            time_minutes
            <
            STRONG_FAULT_END_MIN
        )
    )

    early_gnss_outage = (
        (
            time_minutes
            >=
            EARLY_GNSS_OUTAGE_START_MIN
        )
        &
        (
            time_minutes
            <
            early_gnss_outage_end
        )
    )

    star_tracker_outage = (
        (
            time_minutes
            >=
            STAR_TRACKER_OUTAGE_START_MIN
        )
        &
        (
            time_minutes
            <
            star_tracker_outage_end
        )
    )

    low_redundancy_fault = (
        (
            time_minutes
            >=
            low_fault_start
        )
        &
        (
            time_minutes
            <
            low_fault_end
        )
    )

    post_low_redundancy = (
        (
            time_minutes
            >=
            low_fault_end
        )
        &
        (
            time_minutes
            <
            DUAL_OUTAGE_START_MIN
        )
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
            dual_outage_end
        )
    )

    post_dual_outage = (
        (
            time_minutes
            >=
            dual_outage_end
        )
        &
        (
            time_minutes
            <=
            MISSION_DURATION_MIN
        )
    )

    burn_in = (
        time_minutes
        >=
        BURN_IN_MIN
    )

    return {
        "strong_fault":
            strong_fault,

        "early_gnss_outage":
            early_gnss_outage,

        "star_tracker_outage":
            star_tracker_outage,

        "low_redundancy_fault":
            low_redundancy_fault,

        "post_low_redundancy":
            post_low_redundancy,

        "dual_outage":
            dual_outage,

        "post_dual_outage":
            post_dual_outage,

        "burn_in":
            burn_in,

        "low_fault_start":
            low_fault_start,

        "low_fault_end":
            low_fault_end,

        "dual_outage_end":
            dual_outage_end
    }


# ------------------------------------------------------------
# 7. ATTITUDE MEKF
# ------------------------------------------------------------

def _simulate_attitude(
    scenario,
    event_masks,
    true_attitude
):

    number_of_epochs = (
        len(
            true_attitude[
                "quaternions"
            ]
        )
    )

    gyro_rng = np.random.default_rng(
        int(
            scenario[
                "scenario_seed"
            ]
        )
        +
        101
    )

    star_tracker_rng = np.random.default_rng(
        int(
            scenario[
                "scenario_seed"
            ]
        )
        +
        202
    )

    gyro_noise_std = np.deg2rad(
        float(
            scenario[
                "gyro_noise_std_degps"
            ]
        )
    )

    true_gyro_bias = np.deg2rad(
        np.array(
            [
                scenario[
                    "gyro_bias_x_degps"
                ],

                scenario[
                    "gyro_bias_y_degps"
                ],

                scenario[
                    "gyro_bias_z_degps"
                ]
            ],
            dtype=float
        )
    )

    star_tracker_noise_std = np.deg2rad(
        float(
            scenario[
                "star_tracker_noise_std_deg"
            ]
        )
    )

    gyro_measurements = np.zeros(
        (
            number_of_epochs - 1,
            3
        )
    )

    for index in range(
        number_of_epochs - 1
    ):

        measurement = (
            simulate_gyroscope_measurement(
                true_angular_rate_body=
                    true_attitude[
                        "body_rates"
                    ][
                        index
                    ],

                bias_body=
                    true_gyro_bias,

                noise_std=
                    gyro_noise_std,

                rng=
                    gyro_rng
            )
        )

        gyro_measurements[
            index
        ] = (
            measurement[
                "measurement"
            ]
        )

    quaternion = (
        true_attitude[
            "quaternions"
        ][
            0
        ].copy()
    )

    gyro_bias_estimate = np.zeros(
        3
    )

    initial_attitude_sigma = np.deg2rad(
        0.10
    )

    initial_bias_sigma = np.deg2rad(
        0.005
    )

    covariance = np.diag(
        [
            initial_attitude_sigma**2,
            initial_attitude_sigma**2,
            initial_attitude_sigma**2,

            initial_bias_sigma**2,
            initial_bias_sigma**2,
            initial_bias_sigma**2
        ]
    )

    rotation_history = np.zeros(
        (
            number_of_epochs,
            3,
            3
        )
    )

    attitude_errors = np.zeros(
        number_of_epochs
    )

    attitude_sigma = np.zeros(
        number_of_epochs
    )

    attitude_nis = np.full(
        number_of_epochs,
        np.nan
    )

    rotation_history[
        0
    ] = (
        quaternion_to_rotation_matrix(
            quaternion
        )
    )

    attitude_sigma[
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )

    star_tracker_interval_epochs = int(
        round(
            STAR_TRACKER_PERIOD_SECONDS
            /
            DT
        )
    )

    for index in range(
        1,
        number_of_epochs
    ):

        prediction = (
            predict_attitude_mekf(
                quaternion_body_to_eci=
                    quaternion,

                gyro_bias_body=
                    gyro_bias_estimate,

                covariance=
                    covariance,

                gyro_measurement_body=
                    gyro_measurements[
                        index - 1
                    ],

                dt=
                    DT,

                gyro_noise_std=
                    gyro_noise_std,

                gyro_bias_random_walk_density=
                    GYRO_BIAS_RANDOM_WALK_DENSITY
            )
        )

        quaternion = (
            prediction[
                "quaternion"
            ]
        )

        gyro_bias_estimate = (
            prediction[
                "gyro_bias"
            ]
        )

        covariance = (
            prediction[
                "covariance"
            ]
        )

        star_tracker_available = (
            not event_masks[
                "star_tracker_outage"
            ][
                index
            ]
            and
            not event_masks[
                "dual_outage"
            ][
                index
            ]
        )

        star_tracker_due = (
            index
            %
            star_tracker_interval_epochs
            ==
            0
        )

        if (
            star_tracker_available
            and
            star_tracker_due
        ):

            measurement = (
                simulate_star_tracker_measurement(
                    true_quaternion_body_to_eci=
                        true_attitude[
                            "quaternions"
                        ][
                            index
                        ],

                    attitude_noise_std_rad=
                        star_tracker_noise_std,

                    rng=
                        star_tracker_rng
                )
            )

            update = (
                update_attitude_mekf_with_star_tracker(
                    predicted_quaternion=
                        quaternion,

                    predicted_gyro_bias=
                        gyro_bias_estimate,

                    predicted_covariance=
                        covariance,

                    star_tracker_quaternion=
                        measurement[
                            "measurement"
                        ],

                    star_tracker_noise_std_rad=
                        star_tracker_noise_std
                )
            )

            quaternion = (
                update[
                    "quaternion"
                ]
            )

            gyro_bias_estimate = (
                update[
                    "gyro_bias"
                ]
            )

            covariance = (
                update[
                    "covariance"
                ]
            )

            attitude_nis[
                index
            ] = (
                update[
                    "nis"
                ]
            )

        rotation_history[
            index
        ] = (
            quaternion_to_rotation_matrix(
                quaternion
            )
        )

        attitude_errors[
            index
        ] = (
            quaternion_attitude_error_angle(
                reference_quaternion=
                    true_attitude[
                        "quaternions"
                    ][
                        index
                    ],

                estimated_quaternion=
                    quaternion
            )
        )

        attitude_sigma[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )

    return {
        "rotation_history":
            rotation_history,

        "attitude_errors":
            attitude_errors,

        "attitude_sigma":
            attitude_sigma,

        "attitude_nis":
            attitude_nis,

        "final_gyro_bias_error":
            float(
                np.linalg.norm(
                    gyro_bias_estimate
                    -
                    true_gyro_bias
                )
            )
    }


# ------------------------------------------------------------
# 8. BODY-FRAME ACCELEROMETER
# ------------------------------------------------------------

def _simulate_accelerometer(
    scenario,
    truth,
    true_attitude
):

    number_of_epochs = (
        len(
            truth[
                "time"
            ]
        )
    )

    rng = np.random.default_rng(
        int(
            scenario[
                "scenario_seed"
            ]
        )
        +
        303
    )

    true_bias_body = np.array(
        [
            scenario[
                "accelerometer_bias_x_mps2"
            ],

            scenario[
                "accelerometer_bias_y_mps2"
            ],

            scenario[
                "accelerometer_bias_z_mps2"
            ]
        ],
        dtype=float
    )

    noise_std = float(
        scenario[
            "accelerometer_noise_std_mps2"
        ]
    )

    base_acceleration = float(
        scenario[
            "non_gravitational_acceleration_mps2"
        ]
    )

    measurements = np.zeros(
        (
            number_of_epochs,
            3
        )
    )

    for index in range(
        number_of_epochs
    ):

        true_specific_force = (
            synthetic_drag_like_acceleration(
                time=
                    truth[
                        "time"
                    ][
                        index
                    ],

                position=
                    truth[
                        "truth_states"
                    ][
                        index,
                        0:3
                    ],

                velocity=
                    truth[
                        "truth_states"
                    ][
                        index,
                        3:6
                    ],

                base_acceleration=
                    base_acceleration
            )
        )

        result = (
            simulate_body_frame_accelerometer_measurement(
                true_specific_force_eci=
                    true_specific_force,

                rotation_body_to_eci=
                    true_attitude[
                        "rotations"
                    ][
                        index
                    ],

                bias_body=
                    true_bias_body,

                noise_std=
                    noise_std,

                rng=
                    rng
            )
        )

        measurements[
            index
        ] = (
            result[
                "measurement_body"
            ]
        )

    return measurements


# ------------------------------------------------------------
# 9. GNSS + CLASSICAL FDIR
# ------------------------------------------------------------

def _simulate_gnss_fdir(
    scenario,
    truth,
    event_masks,
    initial_navigation_position
):

    number_of_epochs = (
        len(
            truth[
                "time"
            ]
        )
    )

    rng = np.random.default_rng(
        int(
            scenario[
                "scenario_seed"
            ]
        )
        +
        404
    )

    pseudorange_noise_std = float(
        scenario[
            "pseudorange_noise_std_m"
        ]
    )

    low_fault_magnitude = float(
        scenario[
            "fault_magnitude_m"
        ]
    )

    signal_available = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    integrity_available = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    measurement_accepted = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    fault_detected = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    reconfigured = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    correct_isolation = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    intentional_fault = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    position_measurements = np.full(
        (
            number_of_epochs,
            3
        ),
        np.nan
    )

    position_covariances = np.full(
        (
            number_of_epochs,
            3,
            3
        ),
        np.nan
    )

    satellite_counts = np.zeros(
        number_of_epochs,
        dtype=int
    )

    pdop_history = np.full(
        number_of_epochs,
        np.nan
    )

    previous_position_guess = (
        np.asarray(
            initial_navigation_position,
            dtype=float
        ).copy()
    )

    estimated_clock_bias_meters = (
        0.0
    )

    for index in range(
        1,
        number_of_epochs
    ):

        if (
            event_masks[
                "early_gnss_outage"
            ][
                index
            ]
            or
            event_masks[
                "dual_outage"
            ][
                index
            ]
        ):

            continue

        (
            all_satellite_positions,
            all_satellite_ids
        ) = generate_simplified_gps_positions(
            time_seconds=
                truth[
                    "time"
                ][
                    index
                ]
        )

        (
            _,
            visible_ids
        ) = select_visible_gnss_satellites(
            receiver_position=
                truth[
                    "truth_states"
                ][
                    index,
                    0:3
                ],

            satellite_positions=
                all_satellite_positions,

            satellite_ids=
                all_satellite_ids
        )

        visible_ids = [
            str(
                satellite_id
            ).strip()
            for satellite_id in visible_ids
        ]

        if event_masks[
            "low_redundancy_fault"
        ][
            index
        ]:

            working_ids = (
                _select_five_satellites(
                    visible_ids
                )
            )

            if working_ids is None:

                continue

        else:

            working_ids = (
                visible_ids.copy()
            )

        if len(
            working_ids
        ) < 4:

            continue

        signal_available[
            index
        ] = (
            True
        )

        satellite_counts[
            index
        ] = (
            len(
                working_ids
            )
        )

        integrity_available[
            index
        ] = (
            len(
                working_ids
            )
            >=
            MINIMUM_SATELLITES_FOR_INTEGRITY
        )

        pseudorange_result = (
            simulate_pseudoranges_with_transmit_time(
                receiver_position=
                    truth[
                        "truth_states"
                    ][
                        index,
                        0:3
                    ],

                satellite_ids=
                    working_ids,

                reception_time_seconds=
                    truth[
                        "time"
                    ][
                        index
                    ],

                receiver_clock_bias_seconds=
                    RECEIVER_CLOCK_BIAS_SECONDS,

                pseudorange_noise_std=
                    pseudorange_noise_std,

                rng=
                    rng
            )
        )

        pseudoranges = np.asarray(
            pseudorange_result[
                "pseudoranges"
            ],
            dtype=float
        ).copy()

        fault_active = (
            event_masks[
                "strong_fault"
            ][
                index
            ]
            or
            event_masks[
                "low_redundancy_fault"
            ][
                index
            ]
        )

        intentional_fault[
            index
        ] = (
            fault_active
        )

        actual_fault_target = (
            None
        )

        if fault_active:

            actual_fault_target = (
                _choose_fault_target(
                    working_ids
                )
            )

            fault_index = (
                working_ids.index(
                    actual_fault_target
                )
            )

            if event_masks[
                "strong_fault"
            ][
                index
            ]:

                pseudoranges[
                    fault_index
                ] += (
                    STRONG_FAULT_MAGNITUDE_M
                )

            else:

                pseudoranges[
                    fault_index
                ] += (
                    low_fault_magnitude
                )

        try:

            result = (
                solve_gnss_with_fdir_transmit_time(
                    satellite_ids=
                        working_ids,

                    pseudoranges=
                        pseudoranges,

                    reception_time_seconds=
                        truth[
                            "time"
                        ][
                            index
                        ],

                    initial_position=
                        previous_position_guess,

                    initial_clock_bias_meters=
                        estimated_clock_bias_meters,

                    pseudorange_noise_std=
                        pseudorange_noise_std,

                    confidence=
                        GNSS_CONFIDENCE
                )
            )

        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError
        ):

            continue

        measurement_accepted[
            index
        ] = bool(
            result.get(
                "measurement_accepted",
                False
            )
        )

        fault_detected[
            index
        ] = bool(
            result.get(
                "fault_detected",
                False
            )
        )

        reconfigured[
            index
        ] = bool(
            result.get(
                "reconfigured",
                False
            )
        )

        isolated_id = result.get(
            "isolated_id",
            None
        )

        if (
            fault_active
            and
            isolated_id
            is not None
            and
            actual_fault_target
            is not None
        ):

            correct_isolation[
                index
            ] = (
                str(
                    isolated_id
                ).strip()
                ==
                str(
                    actual_fault_target
                ).strip()
            )

        if not measurement_accepted[
            index
        ]:

            continue

        solution = result.get(
            "solution",
            None
        )

        accepted_satellite_ids = result.get(
            "satellite_ids",
            None
        )

        if (
            solution is None
            or
            accepted_satellite_ids is None
            or
            not solution.get(
                "converged",
                False
            )
        ):

            measurement_accepted[
                index
            ] = (
                False
            )

            continue

        accepted_satellite_ids = [
            str(
                satellite_id
            ).strip()
            for satellite_id in accepted_satellite_ids
        ]

        position = np.asarray(
            solution[
                "position"
            ],
            dtype=float
        )

        estimated_clock_bias_meters = float(
            solution[
                "clock_bias_meters"
            ]
        )

        previous_position_guess = (
            position.copy()
        )

        try:

            covariance_result = (
                compute_transmit_time_solution_covariance(
                    receiver_position=
                        position,

                    satellite_ids=
                        accepted_satellite_ids,

                    reception_time_seconds=
                        truth[
                            "time"
                        ][
                            index
                        ],

                    pseudorange_noise_std=
                        pseudorange_noise_std
                )
            )

        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
            KeyError
        ):

            measurement_accepted[
                index
            ] = (
                False
            )

            continue

        position_measurements[
            index
        ] = (
            position
        )

        position_covariances[
            index
        ] = (
            covariance_result[
                "position_covariance"
            ]
        )

        pdop_history[
            index
        ] = float(
            covariance_result[
                "pdop"
            ]
        )

    return {
        "signal_available":
            signal_available,

        "integrity_available":
            integrity_available,

        "measurement_accepted":
            measurement_accepted,

        "fault_detected":
            fault_detected,

        "reconfigured":
            reconfigured,

        "correct_isolation":
            correct_isolation,

        "intentional_fault":
            intentional_fault,

        "position_measurements":
            position_measurements,

        "position_covariances":
            position_covariances,

        "satellite_counts":
            satellite_counts,

        "pdop":
            pdop_history
    }


# ------------------------------------------------------------
# 10. NAVIGATION BRANCH
# ------------------------------------------------------------

def _run_navigation_branch(
    scenario,
    truth,
    event_masks,
    attitude_result,
    accelerometer_measurements,
    gnss_result,
    protected,
    estimator_start_min=0.0
):

    # ========================================================
    # CONTROLLED NAVIGATION-ESTIMATOR START
    #
    # estimator_start_min = 0.0
    #     reproduces the original AURORA behavior.
    #
    # estimator_start_min > 0.0
    #     keeps the physical spacecraft, truth trajectory,
    #     sensors, attitude solution, GNSS geometry and
    #     disturbances unchanged, but starts the navigation
    #     EKF later in the mission.
    #
    # The delayed EKF receives the SAME initial prior error
    # and SAME initial covariance as the baseline EKF.
    #
    # This makes estimator learning duration the controlled
    # experimental variable.
    # ========================================================

    estimator_start_min = float(
        estimator_start_min
    )


    if (
        not np.isfinite(
            estimator_start_min
        )
        or
        estimator_start_min < 0.0
    ):

        raise ValueError(
            "estimator_start_min must be finite "
            "and >= 0."
        )


    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    time_seconds = np.asarray(
        truth[
            "time"
        ],
        dtype=float
    )


    if (
        "time_minutes"
        in truth
    ):

        time_minutes = np.asarray(
            truth[
                "time_minutes"
            ],
            dtype=float
        )


    else:

        time_minutes = (
            time_seconds
            /
            60.0
        )


    number_of_epochs = len(
        time_seconds
    )


    # --------------------------------------------------------
    # Convert requested start time to simulation epoch.
    #
    # searchsorted(..., side="left") means:
    #
    # if a requested start is not exactly on an epoch,
    # the EKF begins at the first epoch AFTER that time.
    # --------------------------------------------------------

    estimator_start_index = int(
        np.searchsorted(
            time_minutes,
            estimator_start_min,
            side="left"
        )
    )


    if (
        estimator_start_index
        >=
        number_of_epochs
    ):

        raise ValueError(
            "Estimator start time occurs after "
            "the end of the simulation."
        )


    actual_estimator_start_min = float(
        time_minutes[
            estimator_start_index
        ]
    )


    # --------------------------------------------------------
    # Ensure the estimator is available when the dual outage
    # begins.
    # --------------------------------------------------------

    if (
        "dual_outage"
        in event_masks
    ):

        dual_indices = np.where(
            np.asarray(
                event_masks[
                    "dual_outage"
                ],
                dtype=bool
            )
        )[0]


    else:

        dual_indices = np.array(
            [],
            dtype=int
        )


    if len(
        dual_indices
    ) > 0:

        dual_outage_start_index = int(
            dual_indices[
                0
            ]
        )


        dual_outage_start_time_min = float(
            time_minutes[
                dual_outage_start_index
            ]
        )


        if (
            estimator_start_index
            >
            dual_outage_start_index
        ):

            raise ValueError(
                "Navigation estimator starts after "
                "the dual outage has already begun."
            )


        estimator_age_at_dual_outage_min = float(
            (
                time_seconds[
                    dual_outage_start_index
                ]
                -
                time_seconds[
                    estimator_start_index
                ]
            )
            /
            60.0
        )


    else:

        dual_outage_start_index = None

        dual_outage_start_time_min = np.nan

        estimator_age_at_dual_outage_min = np.nan


    # ========================================================
    # INITIAL PRIOR ERROR
    #
    # SAME error vector is used regardless of start time.
    # ========================================================

    initial_position_error = np.array(
        [
            scenario[
                "initial_position_error_x_m"
            ],

            scenario[
                "initial_position_error_y_m"
            ],

            scenario[
                "initial_position_error_z_m"
            ]
        ],
        dtype=float
    )


    initial_velocity_error = np.array(
        [
            scenario[
                "initial_velocity_error_x_mps"
            ],

            scenario[
                "initial_velocity_error_y_mps"
            ],

            scenario[
                "initial_velocity_error_z_mps"
            ]
        ],
        dtype=float
    )


    initial_orbit_error = np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )


    # ========================================================
    # STATE AT ESTIMATOR ACTIVATION
    #
    # For estimator_start_min == 0:
    #     use the original implementation exactly.
    #
    # For delayed start:
    #     initialize around the TRUE orbital state at the
    #     activation epoch, plus the SAME controlled prior
    #     error used by the baseline.
    # ========================================================

    if estimator_start_index == 0:

        reference_orbit_state = np.asarray(
            truth[
                "initial_orbital_state"
            ],
            dtype=float
        )


    else:

        reference_orbit_state = np.asarray(
            truth[
                "truth_states"
            ][
                estimator_start_index,
                0:6
            ],
            dtype=float
        )


    state = np.concatenate(
        (
            reference_orbit_state
            +
            initial_orbit_error,

            np.zeros(
                3
            )
        )
    )


    # ========================================================
    # SAME INITIAL COVARIANCE FOR EVERY MATURITY CONDITION
    # ========================================================

    covariance = np.diag(
        [
            100.0**2,
            100.0**2,
            100.0**2,

            0.10**2,
            0.10**2,
            0.10**2,

            (5.0e-6)**2,
            (5.0e-6)**2,
            (5.0e-6)**2
        ]
    )


    # ========================================================
    # NEW RESILIENCE MANAGER AT ESTIMATOR ACTIVATION
    #
    # This prevents a delayed estimator from inheriting
    # manager history accumulated before it existed.
    # ========================================================

    manager = (
        ResilienceManager(
            attitude_sigma_limit_rad=
                ATTITUDE_SIGMA_LIMIT_RAD,

            recovery_epochs=
                RECOVERY_EPOCHS
        )
    )


    # ========================================================
    # OUTPUT ARRAYS
    #
    # Before estimator activation:
    #
    #     position error = NaN
    #     sigma          = NaN
    #     NIS            = NaN
    #
    # This is intentional. There is no navigation estimate
    # before the navigation EKF exists.
    # ========================================================

    position_errors = np.full(
        number_of_epochs,
        np.nan,
        dtype=float
    )


    position_sigmas = np.full(
        number_of_epochs,
        np.nan,
        dtype=float
    )


    nis_history = np.full(
        number_of_epochs,
        np.nan,
        dtype=float
    )


    update_used = np.zeros(
        number_of_epochs,
        dtype=bool
    )


    estimator_active = np.zeros(
        number_of_epochs,
        dtype=bool
    )


    estimator_active[
        estimator_start_index:
    ] = True


    # ========================================================
    # ESTIMATOR STATE AT ACTIVATION EPOCH
    # ========================================================

    position_errors[
        estimator_start_index
    ] = np.linalg.norm(
        state[
            0:3
        ]
        -
        truth[
            "truth_states"
        ][
            estimator_start_index,
            0:3
        ]
    )


    position_sigmas[
        estimator_start_index
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    accelerometer_noise_std = float(
        scenario[
            "accelerometer_noise_std_mps2"
        ]
    )


    # ========================================================
    # EKF LOOP
    #
    # Original implementation started at index 1.
    #
    # Controlled version starts one epoch AFTER estimator
    # activation, which gives exactly the same behavior when
    # estimator_start_index == 0.
    # ========================================================

    for index in range(
        estimator_start_index + 1,
        number_of_epochs
    ):

        (
            predicted_state,
            predicted_covariance
        ) = (
            ekf_predict_with_body_bias_estimation(
                state=
                    state,

                covariance=
                    covariance,

                dt=
                    DT,

                accelerometer_measurement_body=
                    accelerometer_measurements[
                        index - 1
                    ],

                rotation_body_to_eci=
                    attitude_result[
                        "rotation_history"
                    ][
                        index - 1
                    ],

                accelerometer_noise_std=
                    accelerometer_noise_std,

                bias_random_walk_density=
                    ACCELEROMETER_BIAS_RANDOM_WALK_DENSITY
            )
        )


        if protected:

            integrity_flag = bool(
                gnss_result[
                    "integrity_available"
                ][
                    index
                ]
            )


        else:

            integrity_flag = True


        star_tracker_available = (
            not event_masks[
                "star_tracker_outage"
            ][
                index
            ]
            and
            not event_masks[
                "dual_outage"
            ][
                index
            ]
        )


        decision = (
            manager.update(
                gnss_signal_available=
                    bool(
                        gnss_result[
                            "signal_available"
                        ][
                            index
                        ]
                    ),

                gnss_measurement_accepted=
                    bool(
                        gnss_result[
                            "measurement_accepted"
                        ][
                            index
                        ]
                    ),

                gnss_fault_detected=
                    bool(
                        gnss_result[
                            "fault_detected"
                        ][
                            index
                        ]
                    ),

                gnss_reconfigured=
                    bool(
                        gnss_result[
                            "reconfigured"
                        ][
                            index
                        ]
                    ),

                star_tracker_available=
                    star_tracker_available,

                attitude_sigma_rad=
                    float(
                        attitude_result[
                            "attitude_sigma"
                        ][
                            index
                        ]
                    ),

                gnss_integrity_available=
                    integrity_flag
            )
        )


        measurement_exists = (
            np.all(
                np.isfinite(
                    gnss_result[
                        "position_measurements"
                    ][
                        index
                    ]
                )
            )
            and
            np.all(
                np.isfinite(
                    gnss_result[
                        "position_covariances"
                    ][
                        index
                    ]
                )
            )
        )


        if (
            decision.use_gnss_update
            and
            measurement_exists
        ):

            update = (
                ekf_update_position_augmented(
                    predicted_state=
                        predicted_state,

                    predicted_covariance=
                        predicted_covariance,

                    position_measurement=
                        gnss_result[
                            "position_measurements"
                        ][
                            index
                        ],

                    measurement_noise_covariance=
                        gnss_result[
                            "position_covariances"
                        ][
                            index
                        ]
                )
            )


            innovation = (
                update[
                    "innovation"
                ]
            )


            innovation_covariance = (
                update[
                    "innovation_covariance"
                ]
            )


            nis_history[
                index
            ] = float(
                innovation.T
                @
                np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )


            state = (
                update[
                    "state"
                ]
            )


            covariance = (
                update[
                    "covariance"
                ]
            )


            update_used[
                index
            ] = True


        else:

            state = (
                predicted_state
            )


            covariance = (
                predicted_covariance
            )


        position_errors[
            index
        ] = np.linalg.norm(
            state[
                0:3
            ]
            -
            truth[
                "truth_states"
            ][
                index,
                0:3
            ]
        )


        position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


    # ========================================================
    # NIS CONSISTENCY
    # ========================================================

    valid_nis = np.isfinite(
        nis_history
    )


    nis_lower = chi2.ppf(
        0.025,
        df=3
    )


    nis_upper = chi2.ppf(
        0.975,
        df=3
    )


    if np.any(
        valid_nis
    ):

        nis_mean = float(
            np.mean(
                nis_history[
                    valid_nis
                ]
            )
        )


        nis_coverage = float(
            100.0
            *
            np.mean(
                (
                    nis_history[
                        valid_nis
                    ]
                    >=
                    nis_lower
                )
                &
                (
                    nis_history[
                        valid_nis
                    ]
                    <=
                    nis_upper
                )
            )
        )


    else:

        nis_mean = np.nan
        nis_coverage = np.nan


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "nis":
            nis_history,

        "nis_mean":
            nis_mean,

        "nis_coverage":
            nis_coverage,

        "update_used":
            update_used,

        "estimator_active":
            estimator_active,

        "estimator_start_index":
            int(
                estimator_start_index
            ),

        "estimator_start_time_min":
            float(
                actual_estimator_start_min
            ),

        "estimator_requested_start_min":
            float(
                estimator_start_min
            ),

        "dual_outage_start_time_min":
            float(
                dual_outage_start_time_min
            ),

        "estimator_age_at_dual_outage_min":
            float(
                estimator_age_at_dual_outage_min
            )
    }

# ------------------------------------------------------------
# 11. POLICY METRICS
# ------------------------------------------------------------

def _policy_metrics(
    branch,
    event_masks
):

    errors = (
        branch[
            "position_errors"
        ]
    )

    sigma = (
        branch[
            "position_sigmas"
        ]
    )

    burn_in = (
        event_masks[
            "burn_in"
        ]
    )

    return {
        "mission_rmse":
            _rmse(
                errors[
                    burn_in
                ]
            ),

        "mission_max":
            float(
                np.max(
                    errors[
                        burn_in
                    ]
                )
            ),

        "final_error":
            float(
                errors[
                    -1
                ]
            ),

        "final_sigma":
            float(
                sigma[
                    -1
                ]
            ),

        "strong_fault_rmse":
            _rmse(
                errors[
                    event_masks[
                        "strong_fault"
                    ]
                ]
            ),

        "gnss_outage_rmse":
            _rmse(
                errors[
                    event_masks[
                        "early_gnss_outage"
                    ]
                ]
            ),

        "star_tracker_outage_rmse":
            _rmse(
                errors[
                    event_masks[
                        "star_tracker_outage"
                    ]
                ]
            ),

        "low_redundancy_rmse":
            _rmse(
                errors[
                    event_masks[
                        "low_redundancy_fault"
                    ]
                ]
            ),

        "post_low_rmse":
            _rmse(
                errors[
                    event_masks[
                        "post_low_redundancy"
                    ]
                ]
            ),

        "dual_outage_rmse":
            _rmse(
                errors[
                    event_masks[
                        "dual_outage"
                    ]
                ]
            ),

        "post_dual_rmse":
            _rmse(
                errors[
                    event_masks[
                        "post_dual_outage"
                    ]
                ]
            ),

        "nis_mean":
            float(
                branch[
                    "nis_mean"
                ]
            ),

        "nis_coverage":
            float(
                branch[
                    "nis_coverage"
                ]
            ),

        "updates":
            int(
                np.sum(
                    branch[
                        "update_used"
                    ]
                )
            )
    }


# ------------------------------------------------------------
# 12. MAIN SINGLE-SCENARIO RUNNER
# ------------------------------------------------------------

def run_full_mission_robustness_scenario(
    scenario
):

    scenario = dict(
        scenario
    )

    truth = (
        _build_truth(
            base_acceleration=
                float(
                    scenario[
                        "non_gravitational_acceleration_mps2"
                    ]
                )
        )
    )

    true_attitude = (
        _build_true_attitude(
            truth[
                "truth_states"
            ]
        )
    )

    event_masks = (
        _build_event_masks(
            truth[
                "time_minutes"
            ],
            scenario
        )
    )

    attitude_result = (
        _simulate_attitude(
            scenario=
                scenario,

            event_masks=
                event_masks,

            true_attitude=
                true_attitude
        )
    )

    accelerometer_measurements = (
        _simulate_accelerometer(
            scenario=
                scenario,

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
                scenario[
                    "initial_position_error_x_m"
                ],

                scenario[
                    "initial_position_error_y_m"
                ],

                scenario[
                    "initial_position_error_z_m"
                ]
            ],
            dtype=float
        )
    )

    gnss_result = (
        _simulate_gnss_fdir(
            scenario=
                scenario,

            truth=
                truth,

            event_masks=
                event_masks,

            initial_navigation_position=
                initial_navigation_position
        )
    )

    permissive = (
        _run_navigation_branch(
            scenario=
                scenario,

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
        _run_navigation_branch(
            scenario=
                scenario,

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
        _policy_metrics(
            permissive,
            event_masks
        )
    )

    protected_metrics = (
        _policy_metrics(
            protected,
            event_masks
        )
    )

    # ========================================================
    # FDIR metrics
    # ========================================================

    strong_mask = (
        event_masks[
            "strong_fault"
        ]
        &
        gnss_result[
            "signal_available"
        ]
    )

    low_mask = (
        event_masks[
            "low_redundancy_fault"
        ]
        &
        gnss_result[
            "signal_available"
        ]
    )

    healthy_mask = (
        gnss_result[
            "signal_available"
        ]
        &
        ~gnss_result[
            "intentional_fault"
        ]
    )

    low_missed_mask = (
        low_mask
        &
        ~gnss_result[
            "fault_detected"
        ]
    )

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

    low_detection_rate = (
        _percentage(
            gnss_result[
                "fault_detected"
            ][
                low_mask
            ]
        )
    )

    false_alarm_rate = (
        _percentage(
            gnss_result[
                "fault_detected"
            ][
                healthy_mask
            ]
        )
    )

    permissive_low_update_rate = (
        _percentage(
            permissive[
                "update_used"
            ][
                low_mask
            ]
        )
    )

    protected_low_update_rate = (
        _percentage(
            protected[
                "update_used"
            ][
                low_mask
            ]
        )
    )

    if np.any(
        low_missed_mask
    ):

        permissive_missed_update_rate = (
            _percentage(
                permissive[
                    "update_used"
                ][
                    low_missed_mask
                ]
            )
        )

        protected_missed_update_rate = (
            _percentage(
                protected[
                    "update_used"
                ][
                    low_missed_mask
                ]
            )
        )

    else:

        permissive_missed_update_rate = (
            np.nan
        )

        protected_missed_update_rate = (
            np.nan
        )

    # ========================================================
    # Attitude metrics
    # ========================================================

    attitude_errors_deg = np.rad2deg(
        attitude_result[
            "attitude_errors"
        ]
    )

    valid_attitude_nis = np.isfinite(
        attitude_result[
            "attitude_nis"
        ]
    )

    attitude_nis_mean = (
        _safe_mean(
            attitude_result[
                "attitude_nis"
            ][
                valid_attitude_nis
            ]
        )
    )

    # ========================================================
    # Result row
    # ========================================================

    result = {
        "scenario_id":
            scenario[
                "scenario_id"
            ],

        "scenario_index":
            int(
                scenario[
                    "scenario_index"
                ]
            ),

        "scenario_seed":
            int(
                scenario[
                    "scenario_seed"
                ]
            ),

        "time_sync_error_s":
            float(
                truth[
                    "synchronization_error"
                ]
            ),

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

        "low_fault_missed_epochs":
            int(
                np.sum(
                    low_missed_mask
                )
            ),

        "permissive_low_update_rate":
            permissive_low_update_rate,

        "protected_low_update_rate":
            protected_low_update_rate,

        "permissive_missed_update_rate":
            permissive_missed_update_rate,

        "protected_missed_update_rate":
            protected_missed_update_rate,

        # ----------------------------------------------------
        # ATTITUDE
        # ----------------------------------------------------

        "attitude_rmse_deg":
            _rmse(
                attitude_errors_deg
            ),

        "attitude_st_outage_rmse_deg":
            _rmse(
                attitude_errors_deg[
                    event_masks[
                        "star_tracker_outage"
                    ]
                ]
            ),

        "attitude_dual_outage_rmse_deg":
            _rmse(
                attitude_errors_deg[
                    event_masks[
                        "dual_outage"
                    ]
                ]
            ),

        "attitude_nis_mean":
            attitude_nis_mean,

        "gyro_bias_final_error_degps":
            float(
                np.rad2deg(
                    attitude_result[
                        "final_gyro_bias_error"
                    ]
                )
            ),

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
        # PAIRED GAINS
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
            ),

        "ratio_low_rmse":
            (
                permissive_metrics[
                    "low_redundancy_rmse"
                ]
                /
                max(
                    protected_metrics[
                        "low_redundancy_rmse"
                    ],
                    1.0e-12
                )
            ),

        "ratio_dual_rmse":
            (
                permissive_metrics[
                    "dual_outage_rmse"
                ]
                /
                max(
                    protected_metrics[
                        "dual_outage_rmse"
                    ],
                    1.0e-12
                )
            )
    }

    return result