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
# Experience 013-C — VERSION 2
#
# GNSS INTEGRITY AVAILABILITY
#
# This version deliberately uses the same difficult
# low-redundancy time window identified in 013-B:
#
#       105 -> 115 min
#
# with:
#
#       5 satellites
#       +50 m pseudorange fault
#
#
# Comparison:
#
# 1. PERMISSIVE
#
#       if FDIR misses the fault:
#           use the GNSS update
#
#
# 2. AURORA PROTECTED
#
#       if N_sat < 6:
#           integrity unavailable
#           block GNSS update
#
#
# Important:
#
# We also reproduce the earlier 40-55 min GNSS outage
# so that the GNSS random-number sequence prior to the
# 105-115 min interval is consistent with the 013-B
# end-to-end scenario.
#
# The 20-30 min strong-fault event is also reproduced.
#
# Both navigation policies receive EXACTLY the same
# sensor measurements and FDIR outputs.
# ============================================================


# ------------------------------------------------------------
# 1. TIME
# ------------------------------------------------------------

simulation_duration_minutes = (
    125.0
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
# 2. SCENARIO WINDOWS
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

post_fault_mask = (
    (
        time_minutes
        >= 115.0
    )
    &
    (
        time_minutes
        <= 125.0
    )
)


# ------------------------------------------------------------
# 3. INTEGRITY REQUIREMENT
# ------------------------------------------------------------

minimum_satellites_for_integrity = (
    6
)


# ------------------------------------------------------------
# 4. ORBIT
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
# 8. STAR TRACKER + MEKF
#
# For this experiment the star tracker remains
# continuously available. We isolate GNSS integrity.
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

mekf_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)

mekf_attitude_error_history = np.zeros(
    number_of_epochs
)

mekf_attitude_sigma_history = np.zeros(
    number_of_epochs
)

mekf_nis_history = np.full(
    number_of_epochs,
    np.nan
)

mekf_rotation_history[
    0
] = quaternion_to_rotation_matrix(
    mekf_quaternion
)

mekf_attitude_sigma_history[
    0
] = np.sqrt(
    np.trace(
        mekf_covariance[
            0:3,
            0:3
        ]
    )
)


for index in range(
    1,
    number_of_epochs
):

    prediction = (
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
        prediction[
            "quaternion"
        ]
    )

    mekf_gyro_bias = (
        prediction[
            "gyro_bias"
        ]
    )

    mekf_covariance = (
        prediction[
            "covariance"
        ]
    )


    if (
        index
        %
        star_tracker_interval_epochs
        ==
        0
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
                    mekf_quaternion,

                predicted_gyro_bias=
                    mekf_gyro_bias,

                predicted_covariance=
                    mekf_covariance,

                star_tracker_quaternion=
                    measurement[
                        "measurement"
                    ],

                star_tracker_noise_std_rad=
                    star_tracker_noise_std_rad
            )
        )

        mekf_quaternion = (
            update[
                "quaternion"
            ]
        )

        mekf_gyro_bias = (
            update[
                "gyro_bias"
            ]
        )

        mekf_covariance = (
            update[
                "covariance"
            ]
        )

        mekf_nis_history[
            index
        ] = (
            update[
                "nis"
            ]
        )


    mekf_rotation_history[
        index
    ] = quaternion_to_rotation_matrix(
        mekf_quaternion
    )

    mekf_attitude_error_history[
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

    mekf_attitude_sigma_history[
        index
    ] = np.sqrt(
        np.trace(
            mekf_covariance[
                0:3,
                0:3
            ]
        )
    )


# ------------------------------------------------------------
# 9. ACCELEROMETER
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
# 10. GNSS PARAMETERS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

fault_bias_meters = (
    50.0
)

preferred_fault_satellite = (
    "GPS04"
)

gnss_confidence = (
    0.99
)

gnss_rng = np.random.default_rng(
    2026
)


# ------------------------------------------------------------
# 11. SHARED GNSS / FDIR TIMELINE
# ------------------------------------------------------------

gnss_signal_available = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_measurement_accepted_by_fdir = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_fault_detected = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_reconfigured = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_integrity_available = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_position_measurements = np.full(
    (
        number_of_epochs,
        3
    ),
    np.nan
)

gnss_position_covariances = np.full(
    (
        number_of_epochs,
        3,
        3
    ),
    np.nan
)

working_satellite_count = np.zeros(
    number_of_epochs,
    dtype=int
)

gnss_ls_error = np.full(
    number_of_epochs,
    np.nan
)

pdop_history = np.full(
    number_of_epochs,
    np.nan
)

correct_isolation = np.zeros(
    number_of_epochs,
    dtype=bool
)

intentional_fault_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

estimated_clock_bias_meters = (
    0.0
)

previous_gnss_position_guess = (
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

    # ========================================================
    # REPRODUCE 013-B GNSS OUTAGE
    #
    # Important for matching RNG progression.
    # ========================================================

    if gnss_outage_mask[
        index
    ]:

        continue


    (
        all_satellite_positions,
        all_satellite_ids
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


    working_ids = (
        visible_ids.copy()
    )


    # ========================================================
    # FORCE FIVE SATELLITES ONLY AT 105-115 MIN
    # ========================================================

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


    working_satellite_count[
        index
    ] = len(
        working_ids
    )


    if len(
        working_ids
    ) < 4:

        continue


    gnss_signal_available[
        index
    ] = (
        True
    )


    gnss_integrity_available[
        index
    ] = (
        len(
            working_ids
        )
        >=
        minimum_satellites_for_integrity
    )


    # ========================================================
    # PHYSICAL PSEUDORANGES
    # ========================================================

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


    # ========================================================
    # FAULT INJECTION
    #
    # Reproduce strong 20-30 min event and difficult
    # 5-satellite 105-115 min event.
    # ========================================================

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


    # ========================================================
    # FDIR
    # ========================================================

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
                    previous_gnss_position_guess,

                initial_clock_bias_meters=
                    estimated_clock_bias_meters,

                pseudorange_noise_std=
                    pseudorange_noise_std,

                confidence=
                    gnss_confidence
            )
        )


        gnss_measurement_accepted_by_fdir[
            index
        ] = bool(
            result[
                "measurement_accepted"
            ]
        )


        gnss_fault_detected[
            index
        ] = bool(
            result[
                "fault_detected"
            ]
        )


        gnss_reconfigured[
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
            intentional_fault
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


        # ====================================================
        # STORE FDIR-ACCEPTED SOLUTION
        # ====================================================

        if gnss_measurement_accepted_by_fdir[
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

                position_measurement = (
                    accepted_solution[
                        "position"
                    ]
                )


                estimated_clock_bias_meters = (
                    accepted_solution[
                        "clock_bias_meters"
                    ]
                )


                previous_gnss_position_guess = (
                    position_measurement.copy()
                )


                covariance_result = (
                    compute_transmit_time_solution_covariance(
                        receiver_position=
                            position_measurement,

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


                gnss_position_measurements[
                    index
                ] = (
                    position_measurement
                )


                gnss_position_covariances[
                    index
                ] = (
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


                gnss_ls_error[
                    index
                ] = np.linalg.norm(
                    position_measurement
                    -
                    truth_states[
                        index,
                        0:3
                    ]
                )


            else:

                gnss_measurement_accepted_by_fdir[
                    index
                ] = (
                    False
                )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        gnss_measurement_accepted_by_fdir[
            index
        ] = (
            False
        )


# ------------------------------------------------------------
# 12. NAVIGATION INITIAL CONDITIONS
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

accelerometer_bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 13. NAVIGATION POLICY RUNNER
# ------------------------------------------------------------

def run_navigation_policy(
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

    nis_history = np.full(
        number_of_epochs,
        np.nan
    )

    update_used_history = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    mode_history = []


    # --------------------------------------------------------
    # INITIAL EPOCH
    # --------------------------------------------------------

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
                mekf_attitude_sigma_history[
                    0
                ],

            gnss_integrity_available=
                True
        )
    )


    mode_history.append(
        initial_decision.mode
    )


    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for index in range(
        1,
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
                    dt,

                accelerometer_measurement_body=
                    accelerometer_measurements_body[
                        index - 1
                    ],

                rotation_body_to_eci=
                    mekf_rotation_history[
                        index - 1
                    ],

                accelerometer_noise_std=
                    accelerometer_noise_std,

                bias_random_walk_density=
                    accelerometer_bias_random_walk_density
            )
        )


        if protected_integrity_policy:

            integrity_flag = (
                gnss_integrity_available[
                    index
                ]
            )

        else:

            # Permissive policy ignores the additional
            # integrity-availability requirement.
            integrity_flag = (
                True
            )


        decision = (
            manager.update(
                gnss_signal_available=
                    gnss_signal_available[
                        index
                    ],

                gnss_measurement_accepted=
                    gnss_measurement_accepted_by_fdir[
                        index
                    ],

                gnss_fault_detected=
                    gnss_fault_detected[
                        index
                    ],

                gnss_reconfigured=
                    gnss_reconfigured[
                        index
                    ],

                star_tracker_available=
                    True,

                attitude_sigma_rad=
                    mekf_attitude_sigma_history[
                        index
                    ],

                gnss_integrity_available=
                    integrity_flag
            )
        )


        mode_history.append(
            decision.mode
        )


        measurement_exists = (
            np.all(
                np.isfinite(
                    gnss_position_measurements[
                        index
                    ]
                )
            )
            and
            np.all(
                np.isfinite(
                    gnss_position_covariances[
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
                        gnss_position_measurements[
                            index
                        ],

                    measurement_noise_covariance=
                        gnss_position_covariances[
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


            update_used_history[
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
                ]
            )
        )


    return {
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "nis":
            nis_history,

        "update_used":
            update_used_history,

        "mode_history":
            mode_history
    }


# ------------------------------------------------------------
# 14. RUN POLICIES
# ------------------------------------------------------------

permissive = (
    run_navigation_policy(
        protected_integrity_policy=
            False
    )
)

protected = (
    run_navigation_policy(
        protected_integrity_policy=
            True
    )
)


# ------------------------------------------------------------
# 15. FDIR DIAGNOSTICS
# ------------------------------------------------------------

number_fault_epochs = np.sum(
    low_redundancy_fault_mask
)

number_detections = np.sum(
    gnss_fault_detected[
        low_redundancy_fault_mask
    ]
)

number_missed_detections = (
    number_fault_epochs
    -
    number_detections
)

detection_rate = (
    100.0
    *
    number_detections
    /
    number_fault_epochs
)

miss_rate = (
    100.0
    *
    number_missed_detections
    /
    number_fault_epochs
)


detected_fault_mask = (
    low_redundancy_fault_mask
    &
    gnss_fault_detected
)


missed_fault_mask = (
    low_redundancy_fault_mask
    &
    ~gnss_fault_detected
)


if number_detections > 0:

    rejection_given_detection = (
        100.0
        *
        np.mean(
            ~gnss_measurement_accepted_by_fdir[
                detected_fault_mask
            ]
        )
    )

else:

    rejection_given_detection = (
        np.nan
    )


# ------------------------------------------------------------
# 16. POLICY STATISTICS
# ------------------------------------------------------------

fault_indices = np.where(
    low_redundancy_fault_mask
)[0]

fault_end_index = (
    fault_indices[
        -1
    ]
)


def compute_policy_statistics(
    result
):

    errors = (
        result[
            "position_errors"
        ]
    )

    sigmas = (
        result[
            "position_sigmas"
        ]
    )

    nis = (
        result[
            "nis"
        ]
    )

    updates = (
        result[
            "update_used"
        ]
    )


    valid_nis = np.isfinite(
        nis
    )


    fault_nis_mask = (
        valid_nis
        &
        low_redundancy_fault_mask
    )


    post_nis_mask = (
        valid_nis
        &
        post_fault_mask
    )


    if np.any(
        fault_nis_mask
    ):

        mean_fault_nis = np.mean(
            nis[
                fault_nis_mask
            ]
        )

    else:

        mean_fault_nis = (
            np.nan
        )


    if np.any(
        post_nis_mask
    ):

        mean_post_nis = np.mean(
            nis[
                post_nis_mask
            ]
        )

    else:

        mean_post_nis = (
            np.nan
        )


    return {
        "rmse_fault":
            np.sqrt(
                np.mean(
                    errors[
                        low_redundancy_fault_mask
                    ]**2
                )
            ),

        "end_fault_error":
            errors[
                fault_end_index
            ],

        "end_fault_sigma":
            sigmas[
                fault_end_index
            ],

        "rmse_post":
            np.sqrt(
                np.mean(
                    errors[
                        post_fault_mask
                    ]**2
                )
            ),

        "global_rmse":
            np.sqrt(
                np.mean(
                    errors**2
                )
            ),

        "fault_update_rate":
            100.0
            *
            np.mean(
                updates[
                    low_redundancy_fault_mask
                ]
            ),

        "missed_fault_update_rate":
            (
                100.0
                *
                np.mean(
                    updates[
                        missed_fault_mask
                    ]
                )
                if
                number_missed_detections
                >
                0
                else
                np.nan
            ),

        "mean_fault_nis":
            mean_fault_nis,

        "mean_post_nis":
            mean_post_nis,

        "global_nis":
            np.mean(
                nis[
                    valid_nis
                ]
            )
    }


permissive_stats = (
    compute_policy_statistics(
        permissive
    )
)

protected_stats = (
    compute_policy_statistics(
        protected
    )
)


# ------------------------------------------------------------
# 17. MODE DIAGNOSTICS
# ------------------------------------------------------------

permissive_modes = np.array(
    permissive[
        "mode_history"
    ],
    dtype=object
)

protected_modes = np.array(
    protected[
        "mode_history"
    ],
    dtype=object
)


permissive_nominal_during_missed_fault = (
    (
        100.0
        *
        np.mean(
            permissive_modes[
                missed_fault_mask
            ]
            ==
            ResilienceMode.NOMINAL
        )
    )
    if
    number_missed_detections
    >
    0
    else
    np.nan
)


protected_integrity_coverage = (
    100.0
    *
    np.mean(
        np.array([
            (
                mode
                ==
                ResilienceMode.GNSS_INTEGRITY_UNAVAILABLE
            )
            or
            (
                mode
                ==
                ResilienceMode.GNSS_REJECTED
            )
            for mode in protected[
                "mode_history"
            ]
        ])[
            low_redundancy_fault_mask
        ]
    )
)


# ------------------------------------------------------------
# 18. COMMON STATISTICS
# ------------------------------------------------------------

valid_attitude_nis = np.isfinite(
    mekf_nis_history
)

mean_attitude_nis = np.mean(
    mekf_nis_history[
        valid_attitude_nis
    ]
)

attitude_rmse_deg = np.rad2deg(
    np.sqrt(
        np.mean(
            mekf_attitude_error_history**2
        )
    )
)


# ------------------------------------------------------------
# 19. PERFORMANCE FACTORS
# ------------------------------------------------------------

rmse_improvement_factor = (
    permissive_stats[
        "rmse_fault"
    ]
    /
    protected_stats[
        "rmse_fault"
    ]
)

end_error_improvement_factor = (
    permissive_stats[
        "end_fault_error"
    ]
    /
    protected_stats[
        "end_fault_error"
    ]
)


# ------------------------------------------------------------
# 20. VALIDATION
# ------------------------------------------------------------

scenario_is_discriminating = (
    number_missed_detections
    >
    0
)

protected_blocks_all_low_redundancy = (
    protected_stats[
        "fault_update_rate"
    ]
    <
    1.0e-9
)

permissive_uses_missed_faults = (
    (
        permissive_stats[
            "missed_fault_update_rate"
        ]
        >
        90.0
    )
    if
    scenario_is_discriminating
    else
    False
)

integrity_mode_covers_window = (
    protected_integrity_coverage
    >
    99.0
)

protected_improves_rmse = (
    protected_stats[
        "rmse_fault"
    ]
    <
    permissive_stats[
        "rmse_fault"
    ]
)

protected_improves_end_error = (
    protected_stats[
        "end_fault_error"
    ]
    <
    permissive_stats[
        "end_fault_error"
    ]
)

validation_013c = (
    scenario_is_discriminating
    and
    protected_blocks_all_low_redundancy
    and
    permissive_uses_missed_faults
    and
    integrity_mode_covers_window
    and
    protected_improves_rmse
    and
    protected_improves_end_error
)


# ------------------------------------------------------------
# 21. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "========================================================================================================================================"
)

print(
    "AURORA — Experience 013-C V2"
)

print(
    "Integrity availability sur le scenario faible redondance identifie en 013-B"
)

print(
    "========================================================================================================================================"
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
    f"Scenario faible redondance : "
    f"105.0 a 115.0 min"
)

print(
    f"Satellites pendant scenario : "
    f"5"
)

print(
    f"Minimum pour integrite AURORA : "
    f"{minimum_satellites_for_integrity}"
)

print(
    f"Faute pseudorange : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()


print(
    "----- DETECTABILITE FDIR -----"
)

print(
    f"Detection faute : "
    f"{detection_rate:.2f} %"
)

print(
    f"Faute non detectee : "
    f"{miss_rate:.2f} %"
)

print(
    f"Nombre detections : "
    f"{number_detections}"
)

print(
    f"Nombre non-detections : "
    f"{number_missed_detections}"
)

print(
    f"Rejet conditionnel si detection : "
    f"{rejection_given_detection:.2f} %"
)

print()


print(
    "----- ATTITUDE COMMUNE -----"
)

print(
    f"RMSE MEKF : "
    f"{attitude_rmse_deg:.4f} deg"
)

print(
    f"NIS MEKF : "
    f"{mean_attitude_nis:.3f}"
)

print()


print(
    "----- COMPARAISON POLITIQUES -----"
)

header = (
    f"{'Politique':>18} | "
    f"{'Updates fault':>13} | "
    f"{'Updates misses':>14} | "
    f"{'RMSE fault':>11} | "
    f"{'Err fin':>9} | "
    f"{'Sigma fin':>10} | "
    f"{'RMSE post':>10} | "
    f"{'NIS fault':>10} | "
    f"{'NIS global':>10}"
)

print(
    header
)

print(
    "-" * len(
        header
    )
)

print(
    f"{'Permissive':>18} | "
    f"{permissive_stats['fault_update_rate']:12.2f}% | "
    f"{permissive_stats['missed_fault_update_rate']:13.2f}% | "
    f"{permissive_stats['rmse_fault']:11.3f} | "
    f"{permissive_stats['end_fault_error']:9.3f} | "
    f"{permissive_stats['end_fault_sigma']:10.3f} | "
    f"{permissive_stats['rmse_post']:10.3f} | "
    f"{permissive_stats['mean_fault_nis']:10.3f} | "
    f"{permissive_stats['global_nis']:10.3f}"
)

print(
    f"{'AURORA protected':>18} | "
    f"{protected_stats['fault_update_rate']:12.2f}% | "
    f"{protected_stats['missed_fault_update_rate']:13.2f}% | "
    f"{protected_stats['rmse_fault']:11.3f} | "
    f"{protected_stats['end_fault_error']:9.3f} | "
    f"{protected_stats['end_fault_sigma']:10.3f} | "
    f"{protected_stats['rmse_post']:10.3f} | "
    f"{protected_stats['mean_fault_nis']:10.3f} | "
    f"{protected_stats['global_nis']:10.3f}"
)

print()


print(
    "----- SUPERVISION -----"
)

print(
    f"Missed faults marques NOMINAL — permissive : "
    f"{permissive_nominal_during_missed_fault:.2f} %"
)

print(
    f"Couverture integrity-unavailable/rejected — protected : "
    f"{protected_integrity_coverage:.2f} %"
)

print()


print(
    "----- PERFORMANCE -----"
)

print(
    f"Facteur amelioration RMSE fault : "
    f"{rmse_improvement_factor:.2f}"
)

print(
    f"Facteur amelioration erreur fin : "
    f"{end_error_improvement_factor:.2f}"
)

print()


print(
    "----- VALIDATION -----"
)

print(
    f"Scenario discriminant (au moins une non-detection) : "
    f"{scenario_is_discriminating}"
)

print(
    f"Protected bloque tous les updates Nsat<6 : "
    f"{protected_blocks_all_low_redundancy}"
)

print(
    f"Permissive utilise les fautes non detectees : "
    f"{permissive_uses_missed_faults}"
)

print(
    f"Mode integrite couvre la fenetre : "
    f"{integrity_mode_covers_window}"
)

print(
    f"RMSE protegee meilleure : "
    f"{protected_improves_rmse}"
)

print(
    f"Erreur finale protegee meilleure : "
    f"{protected_improves_end_error}"
)

print()

print(
    f"VALIDATION GLOBALE 013-C V2 : "
    f"{validation_013c}"
)

print(
    "========================================================================================================================================"
)


# ------------------------------------------------------------
# 22. POSITION ERRORS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    time_minutes,
    permissive[
        "position_errors"
    ],
    label="Permissive"
)

plt.plot(
    time_minutes,
    protected[
        "position_errors"
    ],
    label="AURORA protected"
)

plt.axvspan(
    105.0,
    115.0,
    alpha=0.15,
    label="5 satellites + faute"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur position 3D [m]"
)

plt.title(
    "AURORA — Integrity availability : impact navigation"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. ERROR / SIGMA
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    time_minutes,
    permissive[
        "position_errors"
    ],
    label="Erreur permissive"
)

plt.plot(
    time_minutes,
    permissive[
        "position_sigmas"
    ],
    linestyle="--",
    label="Sigma permissive"
)

plt.plot(
    time_minutes,
    protected[
        "position_errors"
    ],
    label="Erreur protected"
)

plt.plot(
    time_minutes,
    protected[
        "position_sigmas"
    ],
    linestyle="--",
    label="Sigma protected"
)

plt.axvspan(
    105.0,
    115.0,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Position [m]"
)

plt.title(
    "AURORA — Erreur et covariance"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. UPDATE DECISIONS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)

plt.step(
    time_minutes,
    permissive[
        "update_used"
    ].astype(
        int
    ),
    where="post",
    label="Permissive"
)

plt.step(
    time_minutes,
    protected[
        "update_used"
    ].astype(
        int
    ),
    where="post",
    label="Protected"
)

plt.axvspan(
    105.0,
    115.0,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Update GNSS utilise"
)

plt.yticks([
    0,
    1
])

plt.title(
    "AURORA — Measurement availability vs integrity availability"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 25. PROTECTED MODES
# ------------------------------------------------------------

mode_to_index = {
    ResilienceMode.NOMINAL:
        0,

    ResilienceMode.RECOVERY:
        1,

    ResilienceMode.GNSS_RECONFIGURED:
        2,

    ResilienceMode.GNSS_INTEGRITY_UNAVAILABLE:
        3,

    ResilienceMode.GNSS_REJECTED:
        4,

    ResilienceMode.GNSS_OUTAGE:
        5,

    ResilienceMode.ATTITUDE_DEGRADED:
        6,

    ResilienceMode.DUAL_OUTAGE:
        7
}

protected_mode_numeric = np.array([
    mode_to_index[
        mode
    ]
    for mode in protected[
        "mode_history"
    ]
])

plt.figure(
    figsize=(13, 6)
)

plt.step(
    time_minutes,
    protected_mode_numeric,
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

plt.axvspan(
    105.0,
    115.0,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Mode AURORA"
)

plt.title(
    "AURORA — Integrity availability supervision"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()