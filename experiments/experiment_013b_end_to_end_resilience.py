import numpy as np
import matplotlib.pyplot as plt

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
    ResilienceManager,
    ResilienceMode
)


# ============================================================
# AURORA
# Experience 013-B
#
# CORRECTED END-TO-END VERSION
#
# Real chain:
#
# pseudoranges
#      |
#      v
# transmit-time GNSS
#      |
#      v
# Phase 9-E FDIR
#      |
#      v
# accepted / reconfigured / rejected
#      |
#      v
# ResilienceManager
#      |
#      v
# Navigation EKF
#
#
# Attitude:
#
# gyro + star tracker
#      |
#      v
# MEKF
#      |
#      v
# R_BI
#      |
#      v
# BODY accelerometer -> navigation EKF
#
#
# IMPORTANT CORRECTION
#
# The Phase 9-E FDIR already returns:
#
#     measurement_accepted
#     fault_detected
#     reconfigured
#     isolated_id
#     solution
#     satellite_ids
#     pseudoranges
#
# Therefore we use its accepted/reconfigured solution
# DIRECTLY.
#
# We do NOT reconstruct the FDIR decision ourselves.
# ============================================================


# ------------------------------------------------------------
# 1. BASIC HELPERS
# ------------------------------------------------------------

def normalize_satellite_id(
    satellite_id
):
    return str(
        satellite_id
    ).strip()


def satellite_ids_equal(
    satellite_id_1,
    satellite_id_2
):
    if (
        satellite_id_1 is None
        or
        satellite_id_2 is None
    ):
        return False

    return (
        normalize_satellite_id(
            satellite_id_1
        )
        ==
        normalize_satellite_id(
            satellite_id_2
        )
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
            / desired_dt
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
    / 60.0
)


# ------------------------------------------------------------
# 3. SCENARIO
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


# ------------------------------------------------------------
# 4. ORBIT
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH
    + 550_000.0
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
# 5. TRUE ATTITUDE
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
# 6. TRUE BODY RATE
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
# 7. GYROSCOPE
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

gyro_rng = np.random.default_rng(
    2027
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


# ------------------------------------------------------------
# 8. ACCELEROMETER
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)

true_accelerometer_bias_body = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])

accelerometer_rng = np.random.default_rng(
    4242
)

accelerometer_measurements_body = np.zeros(
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
                accelerometer_rng
        )
    )

    accelerometer_measurements_body[
        index
    ] = (
        result[
            "measurement_body"
        ]
    )


# ------------------------------------------------------------
# 9. ATTITUDE FILTER PARAMETERS
# ------------------------------------------------------------

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

star_tracker_rng = np.random.default_rng(
    3030
)

mekf_quaternion = (
    true_quaternion_history[
        0
    ].copy()
)

mekf_gyro_bias = np.zeros(
    3
)

initial_attitude_sigma = np.deg2rad(
    0.10
)

initial_gyro_bias_sigma = np.deg2rad(
    0.005
)

mekf_covariance = np.diag([
    initial_attitude_sigma**2,
    initial_attitude_sigma**2,
    initial_attitude_sigma**2,

    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2
])

gyro_bias_random_walk_density = np.deg2rad(
    1.0e-7
)


# ------------------------------------------------------------
# 10. NAVIGATION FILTER
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

navigation_state = np.concatenate(
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

navigation_covariance = np.diag([
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

accelerometer_bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 11. GNSS PARAMETERS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

estimated_clock_bias_meters = (
    0.0
)

gnss_rng = np.random.default_rng(
    2026
)

gnss_confidence = (
    0.99
)

fault_bias_meters = (
    50.0
)

preferred_fault_satellite = (
    "GPS04"
)


# ------------------------------------------------------------
# 12. RESILIENCE MANAGER
# ------------------------------------------------------------

attitude_sigma_limit_deg = (
    0.5
)

manager = (
    ResilienceManager(
        attitude_sigma_limit_rad=
            np.deg2rad(
                attitude_sigma_limit_deg
            ),

        recovery_epochs=
            6
    )
)


# ------------------------------------------------------------
# 13. STORAGE
# ------------------------------------------------------------

position_error_history = np.zeros(
    number_of_epochs
)

position_sigma_history = np.zeros(
    number_of_epochs
)

accelerometer_bias_error_history = np.zeros(
    number_of_epochs
)

attitude_error_history = np.zeros(
    number_of_epochs
)

attitude_sigma_history = np.zeros(
    number_of_epochs
)

gyro_bias_error_history = np.zeros(
    number_of_epochs
)

navigation_nis_history = np.full(
    number_of_epochs,
    np.nan
)

attitude_nis_history = np.full(
    number_of_epochs,
    np.nan
)

gnss_ls_error_history = np.full(
    number_of_epochs,
    np.nan
)

pdop_history = np.full(
    number_of_epochs,
    np.nan
)

visible_satellite_count_history = np.zeros(
    number_of_epochs,
    dtype=int
)

working_satellite_count_history = np.zeros(
    number_of_epochs,
    dtype=int
)

fault_detected_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

fault_reconfigured_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_accepted_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

correct_isolation_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

intentional_fault_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

mode_history = []

severity_history = np.zeros(
    number_of_epochs,
    dtype=int
)

actual_fault_target_history = np.full(
    number_of_epochs,
    "",
    dtype=object
)

isolated_satellite_history = np.full(
    number_of_epochs,
    "",
    dtype=object
)


# ------------------------------------------------------------
# 14. INITIAL DIAGNOSTICS
# ------------------------------------------------------------

position_error_history[
    0
] = np.linalg.norm(
    navigation_state[
        0:3
    ]
    -
    truth_states[
        0,
        0:3
    ]
)

position_sigma_history[
    0
] = np.sqrt(
    np.trace(
        navigation_covariance[
            0:3,
            0:3
        ]
    )
)

accelerometer_bias_error_history[
    0
] = np.linalg.norm(
    navigation_state[
        6:9
    ]
    -
    true_accelerometer_bias_body
)

attitude_sigma_history[
    0
] = np.sqrt(
    np.trace(
        mekf_covariance[
            0:3,
            0:3
        ]
    )
)

gyro_bias_error_history[
    0
] = np.linalg.norm(
    mekf_gyro_bias
    -
    true_gyro_bias_body
)

initial_decision = (
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
            attitude_sigma_history[
                0
            ]
    )
)

mode_history.append(
    initial_decision.mode
)

severity_history[
    0
] = (
    initial_decision.severity
)


# ------------------------------------------------------------
# 15. END-TO-END LOOP
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # ========================================================
    # A. NAVIGATION PREDICTION
    # ========================================================

    rotation_estimate_previous = (
        quaternion_to_rotation_matrix(
            mekf_quaternion
        )
    )

    (
        predicted_navigation_state,
        predicted_navigation_covariance
    ) = (
        ekf_predict_with_body_bias_estimation(
            state=
                navigation_state,

            covariance=
                navigation_covariance,

            dt=
                dt,

            accelerometer_measurement_body=
                accelerometer_measurements_body[
                    index - 1
                ],

            rotation_body_to_eci=
                rotation_estimate_previous,

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                accelerometer_bias_random_walk_density
        )
    )


    # ========================================================
    # B. MEKF PREDICTION
    # ========================================================

    attitude_prediction = (
        predict_attitude_mekf(
            quaternion_body_to_eci=
                mekf_quaternion,

            gyro_bias_body=
                mekf_gyro_bias,

            covariance=
                mekf_covariance,

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

    mekf_quaternion = (
        attitude_prediction[
            "quaternion"
        ]
    )

    mekf_gyro_bias = (
        attitude_prediction[
            "gyro_bias"
        ]
    )

    mekf_covariance = (
        attitude_prediction[
            "covariance"
        ]
    )


    # ========================================================
    # C. STAR TRACKER
    # ========================================================

    star_tracker_healthy = (
        not star_tracker_outage_mask[
            index
        ]
        and
        not dual_outage_mask[
            index
        ]
    )

    star_tracker_measurement_due = (
        index
        %
        star_tracker_interval_epochs
        ==
        0
    )

    if (
        star_tracker_healthy
        and
        star_tracker_measurement_due
    ):

        star_tracker_result = (
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

        attitude_update = (
            update_attitude_mekf_with_star_tracker(
                predicted_quaternion=
                    mekf_quaternion,

                predicted_gyro_bias=
                    mekf_gyro_bias,

                predicted_covariance=
                    mekf_covariance,

                star_tracker_quaternion=
                    star_tracker_result[
                        "measurement"
                    ],

                star_tracker_noise_std_rad=
                    star_tracker_noise_std_rad
            )
        )

        mekf_quaternion = (
            attitude_update[
                "quaternion"
            ]
        )

        mekf_gyro_bias = (
            attitude_update[
                "gyro_bias"
            ]
        )

        mekf_covariance = (
            attitude_update[
                "covariance"
            ]
        )

        attitude_nis_history[
            index
        ] = (
            attitude_update[
                "nis"
            ]
        )


    # ========================================================
    # D. ATTITUDE DIAGNOSTICS
    # ========================================================

    attitude_error_history[
        index
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index
                ],

            estimated_quaternion=
                mekf_quaternion
        )
    )

    attitude_sigma_history[
        index
    ] = np.sqrt(
        np.trace(
            mekf_covariance[
                0:3,
                0:3
            ]
        )
    )

    gyro_bias_error_history[
        index
    ] = np.linalg.norm(
        mekf_gyro_bias
        -
        true_gyro_bias_body
    )


    # ========================================================
    # E. DEFAULT GNSS STATUS
    # ========================================================

    gnss_physical_outage = (
        gnss_outage_mask[
            index
        ]
        or
        dual_outage_mask[
            index
        ]
    )

    gnss_signal_available = (
        not gnss_physical_outage
    )

    gnss_measurement_accepted = (
        False
    )

    gnss_fault_detected = (
        False
    )

    gnss_reconfigured = (
        False
    )

    isolated_satellite_id = (
        None
    )

    final_gnss_position = (
        None
    )

    final_gnss_covariance = (
        None
    )


    # ========================================================
    # F. GNSS + REAL PHASE 9-E FDIR
    # ========================================================

    if gnss_signal_available:

        (
            all_satellite_positions,
            all_satellite_ids
        ) = (
            generate_simplified_gps_positions(
                time_seconds=
                    time[
                        index
                    ]
            )
        )

        (
            _,
            visible_ids
        ) = (
            select_visible_gnss_satellites(
                receiver_position=
                    truth_states[
                        index,
                        0:3
                    ],

                satellite_positions=
                    all_satellite_positions,

                satellite_ids=
                    all_satellite_ids
            )
        )

        visible_ids = [
            normalize_satellite_id(
                satellite_id
            )
            for satellite_id in visible_ids
        ]

        visible_satellite_count_history[
            index
        ] = len(
            visible_ids
        )

        working_ids = (
            visible_ids.copy()
        )


        # ----------------------------------------------------
        # LOW-REDUNDANCY CASE
        #
        # Deliberately keep only five satellites.
        # ----------------------------------------------------

        if (
            low_redundancy_fault_mask[
                index
            ]
            and
            len(
                working_ids
            )
            >
            5
        ):

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


        working_satellite_count_history[
            index
        ] = len(
            working_ids
        )


        if len(
            working_ids
        ) >= 4:

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
                        gnss_rng
                )
            )

            pseudoranges = np.asarray(
                pseudorange_result[
                    "pseudoranges"
                ],
                dtype=float
            ).copy()


            # ------------------------------------------------
            # FAULT INJECTION
            # ------------------------------------------------

            intentional_fault = (
                strong_fault_mask[
                    index
                ]
                or
                low_redundancy_fault_mask[
                    index
                ]
            )

            intentional_fault_history[
                index
            ] = (
                intentional_fault
            )

            actual_fault_target = (
                None
            )

            if intentional_fault:

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

                actual_fault_target_history[
                    index
                ] = (
                    actual_fault_target
                )


            # ------------------------------------------------
            # EXACT FDIR API
            # ------------------------------------------------

            try:

                fdir_result = (
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
                            predicted_navigation_state[
                                0:3
                            ],

                        initial_clock_bias_meters=
                            estimated_clock_bias_meters,

                        pseudorange_noise_std=
                            pseudorange_noise_std,

                        confidence=
                            gnss_confidence
                    )
                )


                # ============================================
                # DIRECT USE OF PHASE 9-E OUTPUT
                # ============================================

                gnss_measurement_accepted = bool(
                    fdir_result[
                        "measurement_accepted"
                    ]
                )

                gnss_fault_detected = bool(
                    fdir_result[
                        "fault_detected"
                    ]
                )

                gnss_reconfigured = bool(
                    fdir_result[
                        "reconfigured"
                    ]
                )

                isolated_satellite_id = (
                    fdir_result[
                        "isolated_id"
                    ]
                )


                fault_detected_history[
                    index
                ] = (
                    gnss_fault_detected
                )

                fault_reconfigured_history[
                    index
                ] = (
                    gnss_reconfigured
                )


                if (
                    isolated_satellite_id
                    is not None
                ):

                    isolated_satellite_history[
                        index
                    ] = (
                        isolated_satellite_id
                    )


                if (
                    intentional_fault
                    and
                    isolated_satellite_id
                    is not None
                ):

                    correct_isolation_history[
                        index
                    ] = (
                        satellite_ids_equal(
                            isolated_satellite_id,
                            actual_fault_target
                        )
                    )


                # ============================================
                # ACCEPTED / RECONFIGURED SOLUTION
                #
                # IMPORTANT:
                #
                # solution + satellite_ids already correspond
                # to the FDIR-accepted configuration.
                # ============================================

                if gnss_measurement_accepted:

                    accepted_solution = (
                        fdir_result[
                            "solution"
                        ]
                    )

                    accepted_satellite_ids = [
                        normalize_satellite_id(
                            satellite_id
                        )
                        for satellite_id in
                        fdir_result[
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

                        final_gnss_position = (
                            accepted_solution[
                                "position"
                            ]
                        )

                        estimated_clock_bias_meters = (
                            accepted_solution[
                                "clock_bias_meters"
                            ]
                        )

                        covariance_result = (
                            compute_transmit_time_solution_covariance(
                                receiver_position=
                                    final_gnss_position,

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

                        final_gnss_covariance = (
                            covariance_result[
                                "position_covariance"
                            ]
                        )

                        pdop_history[
                            index
                        ] = (
                            covariance_result[
                                "pdop"
                            ]
                        )

                        gnss_ls_error_history[
                            index
                        ] = np.linalg.norm(
                            final_gnss_position
                            -
                            truth_states[
                                index,
                                0:3
                            ]
                        )

                    else:

                        gnss_measurement_accepted = (
                            False
                        )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError
            ):

                gnss_measurement_accepted = (
                    False
                )


    # ========================================================
    # G. RESILIENCE MANAGER
    # ========================================================

    decision = (
        manager.update(
            gnss_signal_available=
                gnss_signal_available,

            gnss_measurement_accepted=
                gnss_measurement_accepted,

            gnss_fault_detected=
                gnss_fault_detected,

            gnss_reconfigured=
                gnss_reconfigured,

            star_tracker_available=
                star_tracker_healthy,

            attitude_sigma_rad=
                attitude_sigma_history[
                    index
                ]
        )
    )

    mode_history.append(
        decision.mode
    )

    severity_history[
        index
    ] = (
        decision.severity
    )


    # ========================================================
    # H. NAVIGATION UPDATE
    # ========================================================

    if (
        decision.use_gnss_update
        and
        final_gnss_position
        is not None
        and
        final_gnss_covariance
        is not None
    ):

        navigation_update = (
            ekf_update_position_augmented(
                predicted_state=
                    predicted_navigation_state,

                predicted_covariance=
                    predicted_navigation_covariance,

                position_measurement=
                    final_gnss_position,

                measurement_noise_covariance=
                    final_gnss_covariance
            )
        )

        innovation = (
            navigation_update[
                "innovation"
            ]
        )

        innovation_covariance = (
            navigation_update[
                "innovation_covariance"
            ]
        )

        navigation_nis_history[
            index
        ] = (
            innovation.T
            @
            np.linalg.solve(
                innovation_covariance,
                innovation
            )
        )

        navigation_state = (
            navigation_update[
                "state"
            ]
        )

        navigation_covariance = (
            navigation_update[
                "covariance"
            ]
        )

        gnss_accepted_history[
            index
        ] = (
            True
        )

    else:

        navigation_state = (
            predicted_navigation_state
        )

        navigation_covariance = (
            predicted_navigation_covariance
        )


    # ========================================================
    # I. NAVIGATION DIAGNOSTICS
    # ========================================================

    position_error_history[
        index
    ] = np.linalg.norm(
        navigation_state[
            0:3
        ]
        -
        truth_states[
            index,
            0:3
        ]
    )

    position_sigma_history[
        index
    ] = np.sqrt(
        np.trace(
            navigation_covariance[
                0:3,
                0:3
            ]
        )
    )

    accelerometer_bias_error_history[
        index
    ] = np.linalg.norm(
        navigation_state[
            6:9
        ]
        -
        true_accelerometer_bias_body
    )


# ------------------------------------------------------------
# 16. FDIR STATISTICS
# ------------------------------------------------------------

strong_fault_epochs = (
    strong_fault_mask
    &
    intentional_fault_history
)

number_strong_fault_epochs = np.sum(
    strong_fault_epochs
)

strong_fault_detection_rate = (
    100.0
    *
    np.sum(
        fault_detected_history[
            strong_fault_epochs
        ]
    )
    /
    number_strong_fault_epochs
)

strong_fault_correct_isolation_rate = (
    100.0
    *
    np.sum(
        correct_isolation_history[
            strong_fault_epochs
        ]
    )
    /
    number_strong_fault_epochs
)


low_redundancy_epochs = (
    low_redundancy_fault_mask
    &
    intentional_fault_history
)

number_low_redundancy_epochs = np.sum(
    low_redundancy_epochs
)

low_redundancy_detection_rate = (
    100.0
    *
    np.sum(
        fault_detected_history[
            low_redundancy_epochs
        ]
    )
    /
    number_low_redundancy_epochs
)


detected_low_redundancy_mask = (
    low_redundancy_epochs
    &
    fault_detected_history
)

number_detected_low_redundancy = np.sum(
    detected_low_redundancy_mask
)


if (
    number_detected_low_redundancy
    >
    0
):

    low_redundancy_rejection_given_detection = (
        100.0
        *
        np.sum(
            ~gnss_accepted_history[
                detected_low_redundancy_mask
            ]
        )
        /
        number_detected_low_redundancy
    )

else:

    low_redundancy_rejection_given_detection = (
        np.nan
    )


low_redundancy_wrong_reconfiguration_rate = (
    100.0
    *
    np.sum(
        fault_reconfigured_history[
            low_redundancy_epochs
        ]
    )
    /
    number_low_redundancy_epochs
)


healthy_gnss_mask = (
    ~intentional_fault_history
    &
    ~gnss_outage_mask
    &
    ~dual_outage_mask
)

healthy_gnss_mask[
    0
] = (
    False
)

false_alarm_rate = (
    100.0
    *
    np.sum(
        fault_detected_history[
            healthy_gnss_mask
        ]
    )
    /
    np.sum(
        healthy_gnss_mask
    )
)


# ------------------------------------------------------------
# 17. ESTIMATOR STATISTICS
# ------------------------------------------------------------

valid_navigation_nis = np.isfinite(
    navigation_nis_history
)

mean_navigation_nis = np.mean(
    navigation_nis_history[
        valid_navigation_nis
    ]
)

valid_attitude_nis = np.isfinite(
    attitude_nis_history
)

mean_attitude_nis = np.mean(
    attitude_nis_history[
        valid_attitude_nis
    ]
)

valid_gnss_ls = np.isfinite(
    gnss_ls_error_history
)

gnss_ls_rmse = np.sqrt(
    np.mean(
        gnss_ls_error_history[
            valid_gnss_ls
        ]**2
    )
)

valid_pdop = np.isfinite(
    pdop_history
)

mean_pdop = np.mean(
    pdop_history[
        valid_pdop
    ]
)

global_navigation_rmse = np.sqrt(
    np.mean(
        position_error_history**2
    )
)

global_attitude_rmse = np.rad2deg(
    np.sqrt(
        np.mean(
            attitude_error_history**2
        )
    )
)


# ------------------------------------------------------------
# 18. LOW-REDUNDANCY NAVIGATION DIAGNOSTIC
# ------------------------------------------------------------

low_redundancy_navigation_rmse = np.sqrt(
    np.mean(
        position_error_history[
            low_redundancy_fault_mask
        ]**2
    )
)

low_redundancy_accepted_fraction = (
    100.0
    *
    np.mean(
        gnss_accepted_history[
            low_redundancy_fault_mask
        ]
    )
)


# ------------------------------------------------------------
# 19. PERFORMANCE BY MODE
# ------------------------------------------------------------

all_modes = [
    ResilienceMode.NOMINAL,
    ResilienceMode.GNSS_RECONFIGURED,
    ResilienceMode.GNSS_OUTAGE,
    ResilienceMode.GNSS_REJECTED,
    ResilienceMode.ATTITUDE_DEGRADED,
    ResilienceMode.DUAL_OUTAGE,
    ResilienceMode.RECOVERY
]

mode_statistics = {}

for mode in all_modes:

    mask = np.array([
        current_mode
        ==
        mode
        for current_mode in mode_history
    ])

    count = np.sum(
        mask
    )

    if count > 0:

        mode_statistics[
            mode
        ] = {
            "count":
                int(
                    count
                ),

            "rmse_position":
                np.sqrt(
                    np.mean(
                        position_error_history[
                            mask
                        ]**2
                    )
                ),

            "mean_sigma_position":
                np.mean(
                    position_sigma_history[
                        mask
                    ]
                ),

            "rmse_attitude_deg":
                np.rad2deg(
                    np.sqrt(
                        np.mean(
                            attitude_error_history[
                                mask
                            ]**2
                        )
                    )
                )
        }

    else:

        mode_statistics[
            mode
        ] = {
            "count":
                0,

            "rmse_position":
                np.nan,

            "mean_sigma_position":
                np.nan,

            "rmse_attitude_deg":
                np.nan
        }


# ------------------------------------------------------------
# 20. MODE TRANSITIONS
# ------------------------------------------------------------

transitions = [
    (
        time_minutes[
            0
        ],
        mode_history[
            0
        ]
    )
]

previous_mode = (
    mode_history[
        0
    ]
)

for index in range(
    1,
    number_of_epochs
):

    current_mode = (
        mode_history[
            index
        ]
    )

    if (
        current_mode
        !=
        previous_mode
    ):

        transitions.append(
            (
                time_minutes[
                    index
                ],
                current_mode
            )
        )

        previous_mode = (
            current_mode
        )


# ------------------------------------------------------------
# 21. VALIDATION
#
# Important:
#
# We do NOT require high detection probability with
# only five satellites.
#
# That probability is exactly something this experiment
# is intended to diagnose.
#
# What must hold:
#
# - strong fault is detected
# - strong fault is correctly isolated
# - false alarms remain reasonable
# - when a low-redundancy fault IS detected,
#   it is rejected rather than unsafely reconfigured
# - MEKF remains statistically coherent
# ------------------------------------------------------------

observed_modes = set(
    mode_history
)

core_modes_required = {
    ResilienceMode.NOMINAL,
    ResilienceMode.GNSS_RECONFIGURED,
    ResilienceMode.GNSS_OUTAGE,
    ResilienceMode.ATTITUDE_DEGRADED,
    ResilienceMode.DUAL_OUTAGE,
    ResilienceMode.RECOVERY
}

all_core_modes_observed = (
    core_modes_required.issubset(
        observed_modes
    )
)

strong_fdir_valid = (
    strong_fault_detection_rate
    >
    95.0
    and
    strong_fault_correct_isolation_rate
    >
    95.0
)

low_redundancy_safe_when_detected = (
    (
        number_detected_low_redundancy
        >
        0
    )
    and
    (
        low_redundancy_rejection_given_detection
        >
        95.0
    )
    and
    (
        low_redundancy_wrong_reconfiguration_rate
        <
        5.0
    )
)

attitude_filter_valid = (
    1.5
    <
    mean_attitude_nis
    <
    4.5
)

false_alarm_valid = (
    false_alarm_rate
    <
    3.0
)

end_to_end_core_validated = (
    all_core_modes_observed
    and
    strong_fdir_valid
    and
    low_redundancy_safe_when_detected
    and
    attitude_filter_valid
    and
    false_alarm_valid
)


# ------------------------------------------------------------
# 22. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "=================================================================================================================================="
)

print(
    "AURORA — Experience 013-B CORRECTED"
)

print(
    "FDIR GNSS + MEKF + EKF navigation + ResilienceManager"
)

print(
    "=================================================================================================================================="
)

print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
)

print(
    f"Pas temporel : "
    f"{dt:.1f} s"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()


print(
    "----- FDIR — FORTE REDONDANCE -----"
)

print(
    f"Faute pseudorange : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"Detection forte faute : "
    f"{strong_fault_detection_rate:.2f} %"
)

print(
    f"Isolation correcte forte faute : "
    f"{strong_fault_correct_isolation_rate:.2f} %"
)

print(
    f"Taux fausse alarme : "
    f"{false_alarm_rate:.2f} %"
)

print()


print(
    "----- FDIR — FAIBLE REDONDANCE (5 SATELLITES) -----"
)

print(
    f"Detection faute : "
    f"{low_redundancy_detection_rate:.2f} %"
)

print(
    f"Nombre detections : "
    f"{number_detected_low_redundancy}"
)

print(
    f"Rejet conditionnel si faute detectee : "
    f"{low_redundancy_rejection_given_detection:.2f} %"
)

print(
    f"Reconfiguration incorrecte : "
    f"{low_redundancy_wrong_reconfiguration_rate:.2f} %"
)

print(
    f"Updates GNSS acceptees pendant scenario : "
    f"{low_redundancy_accepted_fraction:.2f} %"
)

print(
    f"RMSE navigation pendant faible redondance : "
    f"{low_redundancy_navigation_rmse:.3f} m"
)

print()


print(
    "----- GNSS / NAVIGATION -----"
)

print(
    f"RMSE GNSS LS accepte : "
    f"{gnss_ls_rmse:.3f} m"
)

print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)

print(
    f"RMSE navigation global : "
    f"{global_navigation_rmse:.3f} m"
)

print(
    f"NIS navigation moyen : "
    f"{mean_navigation_nis:.3f}"
)

print(
    "Valeur theorique NIS navigation ~3"
)

print()


print(
    "----- ATTITUDE -----"
)

print(
    f"RMSE attitude global : "
    f"{global_attitude_rmse:.4f} deg"
)

print(
    f"Erreur biais gyro finale : "
    f"{np.rad2deg(gyro_bias_error_history[-1]):.6f} deg/s"
)

print(
    f"NIS attitude moyen : "
    f"{mean_attitude_nis:.3f}"
)

print(
    "Valeur theorique NIS attitude ~3"
)

print()


print(
    "----- TRANSITIONS -----"
)

for (
    transition_time,
    transition_mode
) in transitions:

    print(
        f"t = "
        f"{transition_time:7.2f} min"
        f"  ->  "
        f"{transition_mode.value}"
    )

print()


print(
    "----- PERFORMANCE PAR MODE -----"
)

header = (
    f"{'Mode':>22} | "
    f"{'Epochs':>7} | "
    f"{'RMSE pos [m]':>13} | "
    f"{'Sigma pos [m]':>13} | "
    f"{'RMSE att [deg]':>14}"
)

print(
    header
)

print(
    "-" * len(
        header
    )
)

for mode in all_modes:

    stats = (
        mode_statistics[
            mode
        ]
    )

    print(
        f"{mode.value:>22} | "
        f"{stats['count']:7d} | "
        f"{stats['rmse_position']:13.3f} | "
        f"{stats['mean_sigma_position']:13.3f} | "
        f"{stats['rmse_attitude_deg']:14.4f}"
    )

print()


print(
    "----- VALIDATION -----"
)

print(
    f"Modes coeur observes : "
    f"{all_core_modes_observed}"
)

print(
    f"FDIR forte redondance valide : "
    f"{strong_fdir_valid}"
)

print(
    f"Politique faible redondance sure SI detection : "
    f"{low_redundancy_safe_when_detected}"
)

print(
    f"Fausse alarme acceptable : "
    f"{false_alarm_valid}"
)

print(
    f"MEKF coherent : "
    f"{attitude_filter_valid}"
)

print()

print(
    f"VALIDATION COEUR END-TO-END 013-B : "
    f"{end_to_end_core_validated}"
)

print(
    "=================================================================================================================================="
)


# ------------------------------------------------------------
# 23. MODE PLOT
# ------------------------------------------------------------

mode_to_index = {
    ResilienceMode.NOMINAL:
        0,

    ResilienceMode.RECOVERY:
        1,

    ResilienceMode.GNSS_RECONFIGURED:
        2,

    ResilienceMode.ATTITUDE_DEGRADED:
        3,

    ResilienceMode.GNSS_OUTAGE:
        4,

    ResilienceMode.GNSS_REJECTED:
        5,

    ResilienceMode.DUAL_OUTAGE:
        6
}

mode_numeric_history = np.array([
    mode_to_index[
        mode
    ]
    for mode in mode_history
])

plt.figure(
    figsize=(14, 6)
)

plt.step(
    time_minutes,
    mode_numeric_history,
    where="post"
)

plt.yticks(
    list(
        mode_to_index.values()
    ),

    [
        mode.value
        for mode in mode_to_index.keys()
    ]
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Mode"
)

plt.title(
    "AURORA — Supervision end-to-end"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. NAVIGATION PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time_minutes,
    position_error_history,
    label="Erreur position 3D"
)

plt.plot(
    time_minutes,
    position_sigma_history,
    linestyle="--",
    label="Sigma position 3D"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Position [m]"
)

plt.title(
    "AURORA — Navigation resiliente"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 25. ATTITUDE PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time_minutes,
    np.rad2deg(
        attitude_error_history
    ),
    label="Erreur attitude"
)

plt.plot(
    time_minutes,
    np.rad2deg(
        attitude_sigma_history
    ),
    linestyle="--",
    label="Sigma attitude 3D"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Attitude [deg]"
)

plt.title(
    "AURORA — MEKF pendant modes de resilience"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. FDIR PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)

plt.step(
    time_minutes,
    fault_detected_history.astype(
        int
    ),
    where="post",
    label="Faute detectee"
)

plt.step(
    time_minutes,
    fault_reconfigured_history.astype(
        int
    ),
    where="post",
    label="Reconfiguration"
)

plt.step(
    time_minutes,
    gnss_accepted_history.astype(
        int
    ),
    where="post",
    label="Update GNSS accepte"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Etat logique"
)

plt.yticks([
    0,
    1
])

plt.title(
    "AURORA — FDIR GNSS"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()