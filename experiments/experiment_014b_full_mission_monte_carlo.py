import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    chi2,
    t as student_t
)

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
# Experience 014-B
#
# Paired Monte Carlo full-mission resilience validation
#
#
# Same mission as 014-A:
#
#   20-30 min
#       strong +50 m GNSS fault
#
#   40-55 min
#       GNSS outage
#
#   65-75 min
#       star tracker outage
#
#   105-115 min
#       exactly 5 GNSS satellites
#       +50 m pseudorange fault
#
#   125-140 min
#       simultaneous GNSS + star tracker outage
#
#
# Architectures:
#
#   1. PERMISSIVE
#      FDIR only
#
#   2. AURORA PROTECTED
#      FDIR + integrity availability
#
#
# Monte Carlo design:
#
#   - deterministic orbital truth
#   - deterministic physical sensor biases
#   - independent measurement noise each run
#   - identical measurements for the two architectures
#
# Therefore comparison is PAIRED.
#
#
# NOTE:
#
# The large Phase 16 Monte Carlo campaign will later vary
# more physical parameters and initial uncertainties.
#
# 014-B is specifically a statistical validation of the
# mission-level integrity-policy effect found in 014-A.
# ============================================================


# ------------------------------------------------------------
# 1. MONTE CARLO SETTINGS
# ------------------------------------------------------------

number_of_runs = (
    20
)

burn_in_minutes = (
    10.0
)


# ------------------------------------------------------------
# 2. TIME
# ------------------------------------------------------------

simulation_duration_minutes = (
    150.0
)

simulation_duration = (
    simulation_duration_minutes
    * 60.0
)

desired_dt = (
    10.0
)

number_of_epochs = (
    int(
        np.floor(
            simulation_duration
            /
            desired_dt
        )
    )
    + 1
)

time = np.linspace(
    0.0,
    simulation_duration,
    number_of_epochs
)

dt = (
    time[1]
    -
    time[0]
)

time_minutes = (
    time
    /
    60.0
)


# ------------------------------------------------------------
# 3. MISSION WINDOWS
# ------------------------------------------------------------

strong_fault_mask = (
    (
        time_minutes
        >= 20.0
    )
    &
    (
        time_minutes
        < 30.0
    )
)

gnss_outage_mask = (
    (
        time_minutes
        >= 40.0
    )
    &
    (
        time_minutes
        < 55.0
    )
)

star_tracker_outage_mask = (
    (
        time_minutes
        >= 65.0
    )
    &
    (
        time_minutes
        < 75.0
    )
)

low_redundancy_fault_mask = (
    (
        time_minutes
        >= 105.0
    )
    &
    (
        time_minutes
        < 115.0
    )
)

post_low_redundancy_mask = (
    (
        time_minutes
        >= 115.0
    )
    &
    (
        time_minutes
        < 125.0
    )
)

dual_outage_mask = (
    (
        time_minutes
        >= 125.0
    )
    &
    (
        time_minutes
        < 140.0
    )
)

post_dual_outage_mask = (
    (
        time_minutes
        >= 140.0
    )
    &
    (
        time_minutes
        <= 150.0
    )
)

analysis_mask = (
    time_minutes
    >= burn_in_minutes
)


# ------------------------------------------------------------
# 4. GNSS INTEGRITY REQUIREMENT
# ------------------------------------------------------------

minimum_satellites_for_integrity = (
    6
)

fault_bias_meters = (
    50.0
)

preferred_fault_satellite = (
    "GPS04"
)

pseudorange_noise_std = (
    3.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

gnss_confidence = (
    0.99
)


# ------------------------------------------------------------
# 5. ORBITAL TRUTH
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH
    +
    550_000.0
)

eccentricity = (
    0.01
)

inclination = np.deg2rad(
    97.6
)

raan = np.deg2rad(
    40.0
)

argument_of_periapsis = np.deg2rad(
    30.0
)

true_anomaly = np.deg2rad(
    25.0
)

(
    true_initial_position,
    true_initial_velocity
) = keplerian_to_cartesian(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly
)

true_initial_orbital_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)

true_non_gravitational_acceleration = (
    2.0e-5
)

truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_orbital_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            true_non_gravitational_acceleration
    )
)

truth_states = (
    truth_solution.y.T
)

maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        -
        time
    )
)


# ------------------------------------------------------------
# 6. TRUE ATTITUDE
# ------------------------------------------------------------

true_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)

true_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)

for index in range(
    number_of_epochs
):

    true_rotation_history[
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

    true_quaternion_history[
        index
    ] = (
        rotation_matrix_to_quaternion(
            true_rotation_history[
                index
            ]
        )
    )


# ------------------------------------------------------------
# 7. TRUE BODY RATE
# ------------------------------------------------------------

true_body_rate_history = np.zeros(
    (
        number_of_epochs - 1,
        3
    )
)

for index in range(
    number_of_epochs - 1
):

    relative_rotation = (
        true_rotation_history[
            index
        ].T
        @
        true_rotation_history[
            index + 1
        ]
    )

    rotation_vector = (
        rotation_matrix_to_rotation_vector(
            relative_rotation
        )
    )

    true_body_rate_history[
        index
    ] = (
        rotation_vector
        /
        dt
    )


# ------------------------------------------------------------
# 8. SENSOR PARAMETERS
# ------------------------------------------------------------

true_gyro_bias_deg_per_second = np.array([
    0.0010,
    -0.0008,
    0.0006
])

true_gyro_bias_body = np.deg2rad(
    true_gyro_bias_deg_per_second
)

gyro_noise_std_deg_per_second = (
    0.0005
)

gyro_noise_std = np.deg2rad(
    gyro_noise_std_deg_per_second
)

gyro_bias_random_walk_density = np.deg2rad(
    1.0e-7
)

star_tracker_period_seconds = (
    60.0
)

star_tracker_interval_epochs = int(
    round(
        star_tracker_period_seconds
        /
        dt
    )
)

star_tracker_noise_std_deg = (
    0.02
)

star_tracker_noise_std_rad = np.deg2rad(
    star_tracker_noise_std_deg
)

accelerometer_noise_std = (
    5.0e-7
)

true_accelerometer_bias_body = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])

accelerometer_bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 9. NAVIGATION INITIAL CONDITIONS
# ------------------------------------------------------------

initial_position_error = np.array([
    100.0,
    -80.0,
    60.0
])

initial_velocity_error = np.array([
    0.10,
    -0.08,
    0.05
])

initial_navigation_state = np.concatenate(
    (
        true_initial_orbital_state
        +
        np.concatenate(
            (
                initial_position_error,
                initial_velocity_error
            )
        ),

        np.zeros(
            3
        )
    )
)

initial_navigation_covariance = np.diag([
    100.0**2,
    100.0**2,
    100.0**2,

    0.10**2,
    0.10**2,
    0.10**2,

    (5.0e-6)**2,
    (5.0e-6)**2,
    (5.0e-6)**2
])


# ------------------------------------------------------------
# 10. HELPER — ATTITUDE SIMULATION
# ------------------------------------------------------------

def simulate_attitude_mekf_for_run(
    run_index
):

    gyro_rng = np.random.default_rng(
        100_000
        +
        run_index
    )

    star_tracker_rng = np.random.default_rng(
        200_000
        +
        run_index
    )


    # --------------------------------------------------------
    # Gyro measurements
    # --------------------------------------------------------

    gyro_measurements = np.zeros(
        (
            number_of_epochs - 1,
            3
        )
    )

    for index in range(
        number_of_epochs - 1
    ):

        result = (
            simulate_gyroscope_measurement(
                true_angular_rate_body=
                    true_body_rate_history[
                        index
                    ],

                bias_body=
                    true_gyro_bias_body,

                noise_std=
                    gyro_noise_std,

                rng=
                    gyro_rng
            )
        )

        gyro_measurements[
            index
        ] = (
            result[
                "measurement"
            ]
        )


    # --------------------------------------------------------
    # Initial MEKF
    # --------------------------------------------------------

    quaternion = (
        true_quaternion_history[
            0
        ].copy()
    )

    gyro_bias = np.zeros(
        3
    )

    initial_attitude_sigma = np.deg2rad(
        0.10
    )

    initial_gyro_bias_sigma = np.deg2rad(
        0.005
    )

    covariance = np.diag([
        initial_attitude_sigma**2,
        initial_attitude_sigma**2,
        initial_attitude_sigma**2,

        initial_gyro_bias_sigma**2,
        initial_gyro_bias_sigma**2,
        initial_gyro_bias_sigma**2
    ])


    rotation_history = np.zeros(
        (
            number_of_epochs,
            3,
            3
        )
    )

    attitude_error_history = np.zeros(
        number_of_epochs
    )

    attitude_sigma_history = np.zeros(
        number_of_epochs
    )

    attitude_nis_history = np.full(
        number_of_epochs,
        np.nan
    )


    rotation_history[
        0
    ] = quaternion_to_rotation_matrix(
        quaternion
    )

    attitude_sigma_history[
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    # --------------------------------------------------------
    # MEKF loop
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        prediction = (
            predict_attitude_mekf(
                quaternion_body_to_eci=
                    quaternion,

                gyro_bias_body=
                    gyro_bias,

                covariance=
                    covariance,

                gyro_measurement_body=
                    gyro_measurements[
                        index - 1
                    ],

                dt=
                    dt,

                gyro_noise_std=
                    gyro_noise_std,

                gyro_bias_random_walk_density=
                    gyro_bias_random_walk_density
            )
        )

        quaternion = (
            prediction[
                "quaternion"
            ]
        )

        gyro_bias = (
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
            not star_tracker_outage_mask[
                index
            ]
            and
            not dual_outage_mask[
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
                        true_quaternion_history[
                            index
                        ],

                    attitude_noise_std_rad=
                        star_tracker_noise_std_rad,

                    rng=
                        star_tracker_rng
                )
            )

            update = (
                update_attitude_mekf_with_star_tracker(
                    predicted_quaternion=
                        quaternion,

                    predicted_gyro_bias=
                        gyro_bias,

                    predicted_covariance=
                        covariance,

                    star_tracker_quaternion=
                        measurement[
                            "measurement"
                        ],

                    star_tracker_noise_std_rad=
                        star_tracker_noise_std_rad
                )
            )

            quaternion = (
                update[
                    "quaternion"
                ]
            )

            gyro_bias = (
                update[
                    "gyro_bias"
                ]
            )

            covariance = (
                update[
                    "covariance"
                ]
            )

            attitude_nis_history[
                index
            ] = (
                update[
                    "nis"
                ]
            )


        rotation_history[
            index
        ] = quaternion_to_rotation_matrix(
            quaternion
        )

        attitude_error_history[
            index
        ] = (
            quaternion_attitude_error_angle(
                reference_quaternion=
                    true_quaternion_history[
                        index
                    ],

                estimated_quaternion=
                    quaternion
            )
        )

        attitude_sigma_history[
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

        "attitude_error":
            attitude_error_history,

        "attitude_sigma":
            attitude_sigma_history,

        "attitude_nis":
            attitude_nis_history,

        "final_gyro_bias_error":
            np.linalg.norm(
                gyro_bias
                -
                true_gyro_bias_body
            )
    }


# ------------------------------------------------------------
# 11. HELPER — ACCELEROMETER SIMULATION
# ------------------------------------------------------------

def simulate_accelerometer_for_run(
    run_index
):

    rng = np.random.default_rng(
        300_000
        +
        run_index
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

        true_specific_force_eci = (
            synthetic_drag_like_acceleration(
                time=
                    time[
                        index
                    ],

                position=
                    truth_states[
                        index,
                        0:3
                    ],

                velocity=
                    truth_states[
                        index,
                        3:6
                    ],

                base_acceleration=
                    true_non_gravitational_acceleration
            )
        )

        result = (
            simulate_body_frame_accelerometer_measurement(
                true_specific_force_eci=
                    true_specific_force_eci,

                rotation_body_to_eci=
                    true_rotation_history[
                        index
                    ],

                bias_body=
                    true_accelerometer_bias_body,

                noise_std=
                    accelerometer_noise_std,

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
# 12. HELPER — GNSS / FDIR SIMULATION
# ------------------------------------------------------------

def simulate_gnss_fdir_for_run(
    run_index
):

    rng = np.random.default_rng(
        400_000
        +
        run_index
    )


    signal_available = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    integrity_available = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    fdir_accepted = np.zeros(
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


    estimated_clock_bias_meters = (
        0.0
    )

    previous_position_guess = (
        true_initial_position
        +
        np.array([
            1000.0,
            -800.0,
            600.0
        ])
    )


    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # Physical outages
        # ----------------------------------------------------

        if (
            gnss_outage_mask[
                index
            ]
            or
            dual_outage_mask[
                index
            ]
        ):

            continue


        # ----------------------------------------------------
        # Visible satellites
        # ----------------------------------------------------

        (
            all_positions,
            all_ids
        ) = generate_simplified_gps_positions(
            time_seconds=
                time[
                    index
                ]
        )

        (
            _,
            visible_ids
        ) = select_visible_gnss_satellites(
            receiver_position=
                truth_states[
                    index,
                    0:3
                ],

            satellite_positions=
                all_positions,

            satellite_ids=
                all_ids
        )

        visible_ids = [
            str(
                satellite_id
            ).strip()
            for satellite_id in visible_ids
        ]

        working_ids = (
            visible_ids.copy()
        )


        # ----------------------------------------------------
        # Force 5 satellites in low redundancy window
        # ----------------------------------------------------

        if low_redundancy_fault_mask[
            index
        ]:

            if (
                preferred_fault_satellite
                in
                working_ids
            ):

                other_ids = [
                    satellite_id
                    for satellite_id in working_ids
                    if satellite_id
                    !=
                    preferred_fault_satellite
                ]

                working_ids = (
                    [
                        preferred_fault_satellite
                    ]
                    +
                    other_ids[
                        :4
                    ]
                )

            else:

                working_ids = (
                    working_ids[
                        :5
                    ]
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

        integrity_available[
            index
        ] = (
            len(
                working_ids
            )
            >=
            minimum_satellites_for_integrity
        )


        # ----------------------------------------------------
        # Pseudoranges
        # ----------------------------------------------------

        pseudorange_result = (
            simulate_pseudoranges_with_transmit_time(
                receiver_position=
                    truth_states[
                        index,
                        0:3
                    ],

                satellite_ids=
                    working_ids,

                reception_time_seconds=
                    time[
                        index
                    ],

                receiver_clock_bias_seconds=
                    receiver_clock_bias_seconds,

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


        # ----------------------------------------------------
        # Fault injection
        # ----------------------------------------------------

        fault_active = (
            strong_fault_mask[
                index
            ]
            or
            low_redundancy_fault_mask[
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

            if (
                preferred_fault_satellite
                in
                working_ids
            ):

                actual_fault_target = (
                    preferred_fault_satellite
                )

            else:

                actual_fault_target = (
                    working_ids[
                        0
                    ]
                )

            fault_index = (
                working_ids.index(
                    actual_fault_target
                )
            )

            pseudoranges[
                fault_index
            ] += (
                fault_bias_meters
            )


        # ----------------------------------------------------
        # FDIR
        # ----------------------------------------------------

        try:

            result = (
                solve_gnss_with_fdir_transmit_time(
                    satellite_ids=
                        working_ids,

                    pseudoranges=
                        pseudoranges,

                    reception_time_seconds=
                        time[
                            index
                        ],

                    initial_position=
                        previous_position_guess,

                    initial_clock_bias_meters=
                        estimated_clock_bias_meters,

                    pseudorange_noise_std=
                        pseudorange_noise_std,

                    confidence=
                        gnss_confidence
                )
            )


            fdir_accepted[
                index
            ] = bool(
                result[
                    "measurement_accepted"
                ]
            )

            fault_detected[
                index
            ] = bool(
                result[
                    "fault_detected"
                ]
            )

            reconfigured[
                index
            ] = bool(
                result[
                    "reconfigured"
                ]
            )


            isolated_id = (
                result[
                    "isolated_id"
                ]
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


            # ------------------------------------------------
            # Accepted solution
            # ------------------------------------------------

            if fdir_accepted[
                index
            ]:

                accepted_solution = (
                    result[
                        "solution"
                    ]
                )

                accepted_satellite_ids = [
                    str(
                        satellite_id
                    ).strip()
                    for satellite_id in
                    result[
                        "satellite_ids"
                    ]
                ]


                if (
                    accepted_solution
                    is not None
                    and
                    accepted_solution[
                        "converged"
                    ]
                    and
                    len(
                        accepted_satellite_ids
                    )
                    >= 4
                ):

                    position = (
                        accepted_solution[
                            "position"
                        ]
                    )

                    estimated_clock_bias_meters = (
                        accepted_solution[
                            "clock_bias_meters"
                        ]
                    )

                    previous_position_guess = (
                        position.copy()
                    )


                    covariance_result = (
                        compute_transmit_time_solution_covariance(
                            receiver_position=
                                position,

                            satellite_ids=
                                accepted_satellite_ids,

                            reception_time_seconds=
                                time[
                                    index
                                ],

                            pseudorange_noise_std=
                                pseudorange_noise_std
                        )
                    )


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


                else:

                    fdir_accepted[
                        index
                    ] = (
                        False
                    )


        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError
        ):

            fdir_accepted[
                index
            ] = (
                False
            )


    return {
        "signal_available":
            signal_available,

        "integrity_available":
            integrity_available,

        "fdir_accepted":
            fdir_accepted,

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
            position_covariances
    }


# ------------------------------------------------------------
# 13. HELPER — NAVIGATION POLICY
# ------------------------------------------------------------

def run_navigation_policy(
    accelerometer_measurements,
    attitude_result,
    gnss_result,
    protected_integrity_policy
):

    state = (
        initial_navigation_state.copy()
    )

    covariance = (
        initial_navigation_covariance.copy()
    )


    manager = (
        ResilienceManager(
            attitude_sigma_limit_rad=
                np.deg2rad(
                    0.5
                ),

            recovery_epochs=
                6
        )
    )


    position_errors = np.zeros(
        number_of_epochs
    )

    position_sigmas = np.zeros(
        number_of_epochs
    )

    navigation_nis = np.full(
        number_of_epochs,
        np.nan
    )

    update_used = np.zeros(
        number_of_epochs,
        dtype=bool
    )


    position_errors[
        0
    ] = np.linalg.norm(
        state[
            0:3
        ]
        -
        truth_states[
            0,
            0:3
        ]
    )

    position_sigmas[
        0
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    manager.update(
        gnss_signal_available=
            True,

        gnss_measurement_accepted=
            True,

        gnss_fault_detected=
            False,

        gnss_reconfigured=
            False,

        star_tracker_available=
            True,

        attitude_sigma_rad=
            attitude_result[
                "attitude_sigma"
            ][
                0
            ],

        gnss_integrity_available=
            True
    )


    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

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
                    dt,

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
                    accelerometer_bias_random_walk_density
            )
        )


        # ----------------------------------------------------
        # Integrity policy
        # ----------------------------------------------------

        if protected_integrity_policy:

            integrity_flag = (
                gnss_result[
                    "integrity_available"
                ][
                    index
                ]
            )

        else:

            integrity_flag = (
                True
            )


        star_tracker_available = (
            not star_tracker_outage_mask[
                index
            ]
            and
            not dual_outage_mask[
                index
            ]
        )


        decision = (
            manager.update(
                gnss_signal_available=
                    gnss_result[
                        "signal_available"
                    ][
                        index
                    ],

                gnss_measurement_accepted=
                    gnss_result[
                        "fdir_accepted"
                    ][
                        index
                    ],

                gnss_fault_detected=
                    gnss_result[
                        "fault_detected"
                    ][
                        index
                    ],

                gnss_reconfigured=
                    gnss_result[
                        "reconfigured"
                    ][
                        index
                    ],

                star_tracker_available=
                    star_tracker_available,

                attitude_sigma_rad=
                    attitude_result[
                        "attitude_sigma"
                    ][
                        index
                    ],

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


        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

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


            navigation_nis[
                index
            ] = (
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
            ] = (
                True
            )


        else:

            state = (
                predicted_state
            )

            covariance = (
                predicted_covariance
            )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        position_errors[
            index
        ] = np.linalg.norm(
            state[
                0:3
            ]
            -
            truth_states[
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
            ])
        )


    return {
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "navigation_nis":
            navigation_nis,

        "update_used":
            update_used
    }


# ------------------------------------------------------------
# 14. METRIC HELPERS
# ------------------------------------------------------------

def rmse(
    values
):

    values = np.asarray(
        values
    )

    return np.sqrt(
        np.mean(
            values**2
        )
    )


def mean_std(
    values
):

    values = np.asarray(
        values
    )

    return (
        np.mean(
            values
        ),
        np.std(
            values,
            ddof=1
        )
    )


def paired_difference_confidence_interval(
    permissive_values,
    protected_values,
    confidence=0.95
):

    permissive_values = np.asarray(
        permissive_values
    )

    protected_values = np.asarray(
        protected_values
    )

    differences = (
        permissive_values
        -
        protected_values
    )

    number = len(
        differences
    )

    mean_difference = np.mean(
        differences
    )

    if number < 2:

        return (
            mean_difference,
            np.nan,
            np.nan
        )


    standard_error = (
        np.std(
            differences,
            ddof=1
        )
        /
        np.sqrt(
            number
        )
    )

    alpha = (
        1.0
        -
        confidence
    )

    critical_value = (
        student_t.ppf(
            1.0
            -
            alpha
            /
            2.0,

            df=
                number
                -
                1
        )
    )


    lower = (
        mean_difference
        -
        critical_value
        *
        standard_error
    )

    upper = (
        mean_difference
        +
        critical_value
        *
        standard_error
    )


    return (
        mean_difference,
        lower,
        upper
    )


def policy_metrics(
    result
):

    position_errors = (
        result[
            "position_errors"
        ]
    )

    nis = (
        result[
            "navigation_nis"
        ]
    )


    valid_nis = (
        np.isfinite(
            nis
        )
        &
        analysis_mask
    )


    individual_lower = (
        chi2.ppf(
            0.025,
            df=3
        )
    )

    individual_upper = (
        chi2.ppf(
            0.975,
            df=3
        )
    )


    if np.any(
        valid_nis
    ):

        mean_nis = np.mean(
            nis[
                valid_nis
            ]
        )

        nis_coverage = (
            100.0
            *
            np.mean(
                (
                    nis[
                        valid_nis
                    ]
                    >=
                    individual_lower
                )
                &
                (
                    nis[
                        valid_nis
                    ]
                    <=
                    individual_upper
                )
            )
        )

    else:

        mean_nis = (
            np.nan
        )

        nis_coverage = (
            np.nan
        )


    return {
        "mission_rmse":
            rmse(
                position_errors[
                    analysis_mask
                ]
            ),

        "mission_max":
            np.max(
                position_errors[
                    analysis_mask
                ]
            ),

        "final_error":
            position_errors[
                -1
            ],

        "strong_fault_rmse":
            rmse(
                position_errors[
                    strong_fault_mask
                ]
            ),

        "gnss_outage_rmse":
            rmse(
                position_errors[
                    gnss_outage_mask
                ]
            ),

        "star_tracker_outage_rmse":
            rmse(
                position_errors[
                    star_tracker_outage_mask
                ]
            ),

        "low_redundancy_rmse":
            rmse(
                position_errors[
                    low_redundancy_fault_mask
                ]
            ),

        "post_low_rmse":
            rmse(
                position_errors[
                    post_low_redundancy_mask
                ]
            ),

        "dual_outage_rmse":
            rmse(
                position_errors[
                    dual_outage_mask
                ]
            ),

        "post_dual_rmse":
            rmse(
                position_errors[
                    post_dual_outage_mask
                ]
            ),

        "mean_nis":
            mean_nis,

        "nis_coverage":
            nis_coverage
    }


# ------------------------------------------------------------
# 15. STORAGE
# ------------------------------------------------------------

metric_names = [
    "mission_rmse",
    "mission_max",
    "final_error",
    "strong_fault_rmse",
    "gnss_outage_rmse",
    "star_tracker_outage_rmse",
    "low_redundancy_rmse",
    "post_low_rmse",
    "dual_outage_rmse",
    "post_dual_rmse",
    "mean_nis",
    "nis_coverage"
]


permissive_metrics = {
    name:
        np.zeros(
            number_of_runs
        )
    for name in metric_names
}


protected_metrics = {
    name:
        np.zeros(
            number_of_runs
        )
    for name in metric_names
}


strong_detection_rates = np.zeros(
    number_of_runs
)

strong_isolation_rates = np.zeros(
    number_of_runs
)

low_detection_rates = np.zeros(
    number_of_runs
)

low_rejection_given_detection = np.full(
    number_of_runs,
    np.nan
)

false_alarm_rates = np.zeros(
    number_of_runs
)

permissive_missed_fault_update_rates = np.full(
    number_of_runs,
    np.nan
)

protected_missed_fault_update_rates = np.full(
    number_of_runs,
    np.nan
)

attitude_rmse_values = np.zeros(
    number_of_runs
)

attitude_nis_values = np.zeros(
    number_of_runs
)

attitude_final_bias_error_values = np.zeros(
    number_of_runs
)


permissive_error_histories = np.zeros(
    (
        number_of_runs,
        number_of_epochs
    )
)

protected_error_histories = np.zeros(
    (
        number_of_runs,
        number_of_epochs
    )
)


# ------------------------------------------------------------
# 16. MONTE CARLO LOOP
# ------------------------------------------------------------

print(
    "\n"
    "================================================================================================================"
)

print(
    "AURORA — Experience 014-B"
)

print(
    f"Paired Monte Carlo full mission — "
    f"{number_of_runs} runs"
)

print(
    "================================================================================================================"
)

print(
    "Progression:"
)


for run_index in range(
    number_of_runs
):

    # ========================================================
    # Simulate shared measurements
    # ========================================================

    attitude_result = (
        simulate_attitude_mekf_for_run(
            run_index=
                run_index
        )
    )

    accelerometer_measurements = (
        simulate_accelerometer_for_run(
            run_index=
                run_index
        )
    )

    gnss_result = (
        simulate_gnss_fdir_for_run(
            run_index=
                run_index
        )
    )


    # ========================================================
    # Run both architectures
    # ========================================================

    permissive_result = (
        run_navigation_policy(
            accelerometer_measurements=
                accelerometer_measurements,

            attitude_result=
                attitude_result,

            gnss_result=
                gnss_result,

            protected_integrity_policy=
                False
        )
    )

    protected_result = (
        run_navigation_policy(
            accelerometer_measurements=
                accelerometer_measurements,

            attitude_result=
                attitude_result,

            gnss_result=
                gnss_result,

            protected_integrity_policy=
                True
        )
    )


    # ========================================================
    # Policy metrics
    # ========================================================

    current_permissive_metrics = (
        policy_metrics(
            permissive_result
        )
    )

    current_protected_metrics = (
        policy_metrics(
            protected_result
        )
    )


    for name in metric_names:

        permissive_metrics[
            name
        ][
            run_index
        ] = (
            current_permissive_metrics[
                name
            ]
        )

        protected_metrics[
            name
        ][
            run_index
        ] = (
            current_protected_metrics[
                name
            ]
        )


    permissive_error_histories[
        run_index
    ] = (
        permissive_result[
            "position_errors"
        ]
    )

    protected_error_histories[
        run_index
    ] = (
        protected_result[
            "position_errors"
        ]
    )


    # ========================================================
    # FDIR metrics
    # ========================================================

    strong_fault_epochs = (
        strong_fault_mask
        &
        gnss_result[
            "intentional_fault"
        ]
    )

    low_fault_epochs = (
        low_redundancy_fault_mask
        &
        gnss_result[
            "intentional_fault"
        ]
    )


    strong_detection_rates[
        run_index
    ] = (
        100.0
        *
        np.mean(
            gnss_result[
                "fault_detected"
            ][
                strong_fault_epochs
            ]
        )
    )


    strong_isolation_rates[
        run_index
    ] = (
        100.0
        *
        np.mean(
            gnss_result[
                "correct_isolation"
            ][
                strong_fault_epochs
            ]
        )
    )


    low_detection_rates[
        run_index
    ] = (
        100.0
        *
        np.mean(
            gnss_result[
                "fault_detected"
            ][
                low_fault_epochs
            ]
        )
    )


    detected_low_fault_epochs = (
        low_fault_epochs
        &
        gnss_result[
            "fault_detected"
        ]
    )


    if np.any(
        detected_low_fault_epochs
    ):

        low_rejection_given_detection[
            run_index
        ] = (
            100.0
            *
            np.mean(
                ~gnss_result[
                    "fdir_accepted"
                ][
                    detected_low_fault_epochs
                ]
            )
        )


    healthy_gnss_epochs = (
        ~gnss_result[
            "intentional_fault"
        ]
        &
        ~gnss_outage_mask
        &
        ~dual_outage_mask
    )

    healthy_gnss_epochs[
        0
    ] = (
        False
    )


    false_alarm_rates[
        run_index
    ] = (
        100.0
        *
        np.mean(
            gnss_result[
                "fault_detected"
            ][
                healthy_gnss_epochs
            ]
        )
    )


    # ========================================================
    # Missed-fault update metrics
    # ========================================================

    missed_low_fault_epochs = (
        low_fault_epochs
        &
        ~gnss_result[
            "fault_detected"
        ]
    )


    if np.any(
        missed_low_fault_epochs
    ):

        permissive_missed_fault_update_rates[
            run_index
        ] = (
            100.0
            *
            np.mean(
                permissive_result[
                    "update_used"
                ][
                    missed_low_fault_epochs
                ]
            )
        )

        protected_missed_fault_update_rates[
            run_index
        ] = (
            100.0
            *
            np.mean(
                protected_result[
                    "update_used"
                ][
                    missed_low_fault_epochs
                ]
            )
        )


    # ========================================================
    # Attitude metrics
    # ========================================================

    attitude_rmse_values[
        run_index
    ] = np.rad2deg(
        rmse(
            attitude_result[
                "attitude_error"
            ]
        )
    )


    valid_attitude_nis = np.isfinite(
        attitude_result[
            "attitude_nis"
        ]
    )


    attitude_nis_values[
        run_index
    ] = np.mean(
        attitude_result[
            "attitude_nis"
        ][
            valid_attitude_nis
        ]
    )


    attitude_final_bias_error_values[
        run_index
    ] = np.rad2deg(
        attitude_result[
            "final_gyro_bias_error"
        ]
    )


    # ========================================================
    # Progress
    # ========================================================

    print(
        f"Run "
        f"{run_index + 1:02d}/{number_of_runs} | "
        f"PD low-red = "
        f"{low_detection_rates[run_index]:6.2f}% | "
        f"RMSE low P/R = "
        f"{permissive_metrics['low_redundancy_rmse'][run_index]:7.3f} / "
        f"{protected_metrics['low_redundancy_rmse'][run_index]:7.3f} m | "
        f"RMSE dual P/R = "
        f"{permissive_metrics['dual_outage_rmse'][run_index]:7.3f} / "
        f"{protected_metrics['dual_outage_rmse'][run_index]:7.3f} m | "
        f"NIS protected = "
        f"{protected_metrics['mean_nis'][run_index]:6.3f}"
    )


# ------------------------------------------------------------
# 17. ENSEMBLE FDIR STATISTICS
# ------------------------------------------------------------

strong_detection_mean, strong_detection_std = (
    mean_std(
        strong_detection_rates
    )
)

strong_isolation_mean, strong_isolation_std = (
    mean_std(
        strong_isolation_rates
    )
)

low_detection_mean, low_detection_std = (
    mean_std(
        low_detection_rates
    )
)

false_alarm_mean, false_alarm_std = (
    mean_std(
        false_alarm_rates
    )
)


valid_low_rejection = np.isfinite(
    low_rejection_given_detection
)

mean_low_rejection = np.mean(
    low_rejection_given_detection[
        valid_low_rejection
    ]
)


# ------------------------------------------------------------
# 18. ENSEMBLE ATTITUDE
# ------------------------------------------------------------

attitude_rmse_mean, attitude_rmse_std = (
    mean_std(
        attitude_rmse_values
    )
)

attitude_nis_mean, attitude_nis_std = (
    mean_std(
        attitude_nis_values
    )
)

attitude_bias_mean, attitude_bias_std = (
    mean_std(
        attitude_final_bias_error_values
    )
)


# ------------------------------------------------------------
# 19. POLICY ENSEMBLE TABLE
# ------------------------------------------------------------

comparison_metric_names = [
    "mission_rmse",
    "mission_max",
    "final_error",
    "strong_fault_rmse",
    "gnss_outage_rmse",
    "star_tracker_outage_rmse",
    "low_redundancy_rmse",
    "post_low_rmse",
    "dual_outage_rmse",
    "post_dual_rmse",
    "mean_nis",
    "nis_coverage"
]


# ------------------------------------------------------------
# 20. PAIRED PERFORMANCE
# ------------------------------------------------------------

(
    mission_difference_mean,
    mission_difference_lower,
    mission_difference_upper
) = paired_difference_confidence_interval(
    permissive_metrics[
        "mission_rmse"
    ],
    protected_metrics[
        "mission_rmse"
    ]
)


(
    low_difference_mean,
    low_difference_lower,
    low_difference_upper
) = paired_difference_confidence_interval(
    permissive_metrics[
        "low_redundancy_rmse"
    ],
    protected_metrics[
        "low_redundancy_rmse"
    ]
)


(
    dual_difference_mean,
    dual_difference_lower,
    dual_difference_upper
) = paired_difference_confidence_interval(
    permissive_metrics[
        "dual_outage_rmse"
    ],
    protected_metrics[
        "dual_outage_rmse"
    ]
)


mission_better_fraction = (
    100.0
    *
    np.mean(
        protected_metrics[
            "mission_rmse"
        ]
        <
        permissive_metrics[
            "mission_rmse"
        ]
    )
)


low_better_fraction = (
    100.0
    *
    np.mean(
        protected_metrics[
            "low_redundancy_rmse"
        ]
        <
        permissive_metrics[
            "low_redundancy_rmse"
        ]
    )
)


dual_better_fraction = (
    100.0
    *
    np.mean(
        protected_metrics[
            "dual_outage_rmse"
        ]
        <
        permissive_metrics[
            "dual_outage_rmse"
        ]
    )
)


# ------------------------------------------------------------
# 21. MISSED FAULT STATISTICS
# ------------------------------------------------------------

runs_with_missed_faults = np.isfinite(
    permissive_missed_fault_update_rates
)

fraction_runs_with_missed_faults = (
    100.0
    *
    np.mean(
        runs_with_missed_faults
    )
)


mean_permissive_missed_update_rate = np.mean(
    permissive_missed_fault_update_rates[
        runs_with_missed_faults
    ]
)


mean_protected_missed_update_rate = np.mean(
    protected_missed_fault_update_rates[
        runs_with_missed_faults
    ]
)


# ------------------------------------------------------------
# 22. VALIDATION CONDITIONS
# ------------------------------------------------------------

strong_fdir_valid = (
    strong_detection_mean
    >
    95.0
    and
    strong_isolation_mean
    >
    95.0
)

false_alarm_valid = (
    false_alarm_mean
    <
    3.0
)

low_redundancy_discriminating = (
    fraction_runs_with_missed_faults
    >
    90.0
    and
    low_detection_mean
    <
    90.0
)

protected_blocks_all_missed_faults = (
    np.nanmax(
        protected_missed_fault_update_rates
    )
    <
    1.0
)

permissive_uses_missed_faults = (
    mean_permissive_missed_update_rate
    >
    90.0
)

protected_low_better = (
    low_better_fraction
    >
    90.0
    and
    low_difference_lower
    >
    0.0
)

protected_dual_better = (
    dual_better_fraction
    >
    75.0
    and
    dual_difference_lower
    >
    0.0
)

protected_mission_better = (
    mission_better_fraction
    >
    75.0
    and
    mission_difference_lower
    >
    0.0
)

protected_nis_consistent = (
    1.5
    <
    np.mean(
        protected_metrics[
            "mean_nis"
        ]
    )
    <
    4.5
)

attitude_nis_consistent = (
    1.5
    <
    attitude_nis_mean
    <
    4.5
)


validation_014b = (
    strong_fdir_valid
    and
    false_alarm_valid
    and
    low_redundancy_discriminating
    and
    protected_blocks_all_missed_faults
    and
    permissive_uses_missed_faults
    and
    protected_low_better
    and
    protected_dual_better
    and
    protected_mission_better
    and
    protected_nis_consistent
    and
    attitude_nis_consistent
)


# ------------------------------------------------------------
# 23. PRINT RESULTS
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================================================================"
)

print(
    "AURORA — Experience 014-B"
)

print(
    "Paired Monte Carlo full-mission resilience validation"
)

print(
    "============================================================================================================================================"
)

print(
    f"Nombre de runs : "
    f"{number_of_runs}"
)

print(
    f"Duree par run : "
    f"{simulation_duration_minutes:.1f} min"
)

print(
    f"Pas temporel : "
    f"{dt:.1f} s"
)

print(
    f"Burn-in exclu des metriques mission : "
    f"{burn_in_minutes:.1f} min"
)

print(
    f"Faute pseudorange : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"Minimum satellites pour integrite : "
    f"{minimum_satellites_for_integrity}"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()


# ------------------------------------------------------------
# 24. FDIR TABLE
# ------------------------------------------------------------

print(
    "----- FDIR ENSEMBLE -----"
)

print(
    f"Detection faute forte : "
    f"{strong_detection_mean:.2f} +/- "
    f"{strong_detection_std:.2f} %"
)

print(
    f"Isolation correcte faute forte : "
    f"{strong_isolation_mean:.2f} +/- "
    f"{strong_isolation_std:.2f} %"
)

print(
    f"Detection faible redondance : "
    f"{low_detection_mean:.2f} +/- "
    f"{low_detection_std:.2f} %"
)

print(
    f"Rejet conditionnel si detection faible redondance : "
    f"{mean_low_rejection:.2f} %"
)

print(
    f"Fausse alarme : "
    f"{false_alarm_mean:.2f} +/- "
    f"{false_alarm_std:.2f} %"
)

print(
    f"Runs avec au moins une faute non detectee : "
    f"{fraction_runs_with_missed_faults:.2f} %"
)

print()


# ------------------------------------------------------------
# 25. ATTITUDE TABLE
# ------------------------------------------------------------

print(
    "----- ATTITUDE ENSEMBLE -----"
)

print(
    f"RMSE attitude : "
    f"{attitude_rmse_mean:.4f} +/- "
    f"{attitude_rmse_std:.4f} deg"
)

print(
    f"NIS attitude : "
    f"{attitude_nis_mean:.3f} +/- "
    f"{attitude_nis_std:.3f}"
)

print(
    f"Erreur biais gyro finale : "
    f"{attitude_bias_mean:.6f} +/- "
    f"{attitude_bias_std:.6f} deg/s"
)

print()


# ------------------------------------------------------------
# 26. POLICY TABLE
# ------------------------------------------------------------

print(
    "----- PERFORMANCE ENSEMBLE -----"
)

header = (
    f"{'Metrique':>26} | "
    f"{'Permissive mean':>16} | "
    f"{'Perm std':>10} | "
    f"{'Protected mean':>16} | "
    f"{'Prot std':>10}"
)

print(
    header
)

print(
    "-" * len(
        header
    )
)


metric_display_names = {
    "mission_rmse":
        "Mission RMSE [m]",

    "mission_max":
        "Mission max [m]",

    "final_error":
        "Final error [m]",

    "strong_fault_rmse":
        "Strong fault RMSE [m]",

    "gnss_outage_rmse":
        "GNSS outage RMSE [m]",

    "star_tracker_outage_rmse":
        "ST outage RMSE [m]",

    "low_redundancy_rmse":
        "Low-red RMSE [m]",

    "post_low_rmse":
        "Post low-red RMSE [m]",

    "dual_outage_rmse":
        "Dual outage RMSE [m]",

    "post_dual_rmse":
        "Post dual RMSE [m]",

    "mean_nis":
        "Navigation NIS",

    "nis_coverage":
        "NIS 95% coverage [%]"
}


for metric_name in comparison_metric_names:

    perm_mean, perm_std = mean_std(
        permissive_metrics[
            metric_name
        ]
    )

    prot_mean, prot_std = mean_std(
        protected_metrics[
            metric_name
        ]
    )

    print(
        f"{metric_display_names[metric_name]:>26} | "
        f"{perm_mean:16.3f} | "
        f"{perm_std:10.3f} | "
        f"{prot_mean:16.3f} | "
        f"{prot_std:10.3f}"
    )


print()


# ------------------------------------------------------------
# 27. INTEGRITY POLICY
# ------------------------------------------------------------

print(
    "----- INTEGRITY POLICY -----"
)

print(
    f"Updates sur fautes manquees — permissive : "
    f"{mean_permissive_missed_update_rate:.2f} %"
)

print(
    f"Updates sur fautes manquees — protected : "
    f"{mean_protected_missed_update_rate:.2f} %"
)

print()


# ------------------------------------------------------------
# 28. PAIRED BENEFIT
# ------------------------------------------------------------

print(
    "----- BENEFICE PAIRE PROTECTED -----"
)

print(
    f"Mission RMSE difference "
    f"(permissive - protected) : "
    f"{mission_difference_mean:.3f} m "
    f"[95% CI "
    f"{mission_difference_lower:.3f}, "
    f"{mission_difference_upper:.3f}]"
)

print(
    f"Protected meilleure mission RMSE dans : "
    f"{mission_better_fraction:.1f} % des runs"
)

print()

print(
    f"Low-red RMSE difference : "
    f"{low_difference_mean:.3f} m "
    f"[95% CI "
    f"{low_difference_lower:.3f}, "
    f"{low_difference_upper:.3f}]"
)

print(
    f"Protected meilleure low-red RMSE dans : "
    f"{low_better_fraction:.1f} % des runs"
)

print()

print(
    f"Dual-outage RMSE difference : "
    f"{dual_difference_mean:.3f} m "
    f"[95% CI "
    f"{dual_difference_lower:.3f}, "
    f"{dual_difference_upper:.3f}]"
)

print(
    f"Protected meilleure dual-outage RMSE dans : "
    f"{dual_better_fraction:.1f} % des runs"
)

print()


# ------------------------------------------------------------
# 29. VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)

print(
    f"FDIR forte redondance valide : "
    f"{strong_fdir_valid}"
)

print(
    f"Fausse alarme acceptable : "
    f"{false_alarm_valid}"
)

print(
    f"Scenario faible redondance discriminant : "
    f"{low_redundancy_discriminating}"
)

print(
    f"Protected bloque toutes les fautes manquees : "
    f"{protected_blocks_all_missed_faults}"
)

print(
    f"Permissive utilise les fautes manquees : "
    f"{permissive_uses_missed_faults}"
)

print(
    f"Gain low-red statistiquement positif : "
    f"{protected_low_better}"
)

print(
    f"Gain dual-outage statistiquement positif : "
    f"{protected_dual_better}"
)

print(
    f"Gain mission statistiquement positif : "
    f"{protected_mission_better}"
)

print(
    f"NIS navigation protected coherent : "
    f"{protected_nis_consistent}"
)

print(
    f"NIS attitude coherent : "
    f"{attitude_nis_consistent}"
)

print()

print(
    f"VALIDATION GLOBALE 014-B : "
    f"{validation_014b}"
)

print(
    "============================================================================================================================================"
)


# ------------------------------------------------------------
# 30. FIGURE — ENSEMBLE MEAN POSITION ERROR
# ------------------------------------------------------------

permissive_mean_error = np.mean(
    permissive_error_histories,
    axis=0
)

protected_mean_error = np.mean(
    protected_error_histories,
    axis=0
)

permissive_p10 = np.percentile(
    permissive_error_histories,
    10.0,
    axis=0
)

permissive_p90 = np.percentile(
    permissive_error_histories,
    90.0,
    axis=0
)

protected_p10 = np.percentile(
    protected_error_histories,
    10.0,
    axis=0
)

protected_p90 = np.percentile(
    protected_error_histories,
    90.0,
    axis=0
)


plt.figure(
    figsize=(15, 7)
)

plt.plot(
    time_minutes,
    permissive_mean_error,
    label="Permissive mean"
)

plt.fill_between(
    time_minutes,
    permissive_p10,
    permissive_p90,
    alpha=0.15,
    label="Permissive P10-P90"
)

plt.plot(
    time_minutes,
    protected_mean_error,
    label="AURORA protected mean"
)

plt.fill_between(
    time_minutes,
    protected_p10,
    protected_p90,
    alpha=0.15,
    label="Protected P10-P90"
)

plt.axvspan(
    105.0,
    115.0,
    alpha=0.10,
    label="5 sats + fault"
)

plt.axvspan(
    125.0,
    140.0,
    alpha=0.10,
    label="Dual outage"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur position 3D [m]"
)

plt.title(
    "AURORA — Monte Carlo full-mission position error"
)

plt.grid(
    True
)

plt.legend(
    ncol=2
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 31. FIGURE — PAIRED LOW-RED RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    permissive_metrics[
        "low_redundancy_rmse"
    ],
    protected_metrics[
        "low_redundancy_rmse"
    ]
)

maximum_low_rmse = max(
    np.max(
        permissive_metrics[
            "low_redundancy_rmse"
        ]
    ),
    np.max(
        protected_metrics[
            "low_redundancy_rmse"
        ]
    )
)

plt.plot(
    [
        0.0,
        maximum_low_rmse
    ],
    [
        0.0,
        maximum_low_rmse
    ],
    linestyle="--",
    label="Equal performance"
)

plt.xlabel(
    "Permissive RMSE [m]"
)

plt.ylabel(
    "Protected RMSE [m]"
)

plt.title(
    "AURORA — Paired low-redundancy RMSE"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 32. FIGURE — PAIRED DUAL OUTAGE RMSE
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    permissive_metrics[
        "dual_outage_rmse"
    ],
    protected_metrics[
        "dual_outage_rmse"
    ]
)

maximum_dual_rmse = max(
    np.max(
        permissive_metrics[
            "dual_outage_rmse"
        ]
    ),
    np.max(
        protected_metrics[
            "dual_outage_rmse"
        ]
    )
)

plt.plot(
    [
        0.0,
        maximum_dual_rmse
    ],
    [
        0.0,
        maximum_dual_rmse
    ],
    linestyle="--",
    label="Equal performance"
)

plt.xlabel(
    "Permissive RMSE [m]"
)

plt.ylabel(
    "Protected RMSE [m]"
)

plt.title(
    "AURORA — Paired dual-outage RMSE"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()