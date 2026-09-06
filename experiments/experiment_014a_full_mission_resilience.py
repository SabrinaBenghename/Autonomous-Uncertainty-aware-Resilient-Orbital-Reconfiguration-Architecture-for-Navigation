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
# Experience 014-A
#
# FULL MISSION RESILIENCE SCENARIO
#
#
# Mission timeline:
#
#   0 - 20 min
#       nominal
#
#  20 - 30 min
#       +50 m GNSS pseudorange fault
#       normal redundancy
#
#  40 - 55 min
#       complete GNSS outage
#
#  65 - 75 min
#       star tracker outage
#
# 105 - 115 min
#       exactly five GNSS satellites
#       +50 m pseudorange fault
#
# 125 - 140 min
#       simultaneous GNSS + star tracker outage
#
# 140 - 150 min
#       recovery
#
#
# Comparison:
#
# 1. PERMISSIVE
#
#       FDIR only.
#
#       If the FDIR does not detect a fault,
#       an available GNSS solution is used.
#
#
# 2. AURORA PROTECTED
#
#       FDIR
#       +
#       integrity availability
#
#       N_sat < 6
#           => GNSS integrity unavailable
#           => GNSS update blocked
#
#
# Important:
#
# Both architectures share EXACTLY:
#
#       truth orbit
#       accelerometer measurements
#       gyro measurements
#       star tracker measurements
#       pseudoranges
#       FDIR outputs
#
# This isolates the effect of the integrity policy.
# ============================================================


# ------------------------------------------------------------
# 1. TIME
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
# 2. MISSION WINDOWS
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
# 9. STAR TRACKER + MEKF
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

mekf_gyro_bias_error_history = np.zeros(
    number_of_epochs
)

mekf_nis_history = np.full(
    number_of_epochs,
    np.nan
)

star_tracker_used_history = np.zeros(
    number_of_epochs,
    dtype=bool
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

mekf_gyro_bias_error_history[
    0
] = np.linalg.norm(
    mekf_gyro_bias
    -
    true_gyro_bias_body
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


    star_tracker_available = (
        not star_tracker_outage_mask[
            index
        ]
        and
        not dual_outage_mask[
            index
        ]
    )


    measurement_due = (
        index
        %
        star_tracker_interval_epochs
        ==
        0
    )


    if (
        star_tracker_available
        and
        measurement_due
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

        star_tracker_used_history[
            index
        ] = (
            True
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

    mekf_gyro_bias_error_history[
        index
    ] = np.linalg.norm(
        mekf_gyro_bias
        -
        true_gyro_bias_body
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

gnss_signal_available_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_integrity_available_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_fdir_accepted_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_fault_detected_history = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_reconfigured_history = np.zeros(
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

working_satellite_count_history = np.zeros(
    number_of_epochs,
    dtype=int
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

gnss_ls_error_history = np.full(
    number_of_epochs,
    np.nan
)

pdop_history = np.full(
    number_of_epochs,
    np.nan
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
    # PHYSICAL GNSS OUTAGES
    # ========================================================

    physical_outage = (
        gnss_outage_mask[
            index
        ]
        or
        dual_outage_mask[
            index
        ]
    )

    if physical_outage:

        continue


    # ========================================================
    # SATELLITE VISIBILITY
    # ========================================================

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
    # LOW REDUNDANCY:
    #
    # force exactly 5 satellites.
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


    working_satellite_count_history[
        index
    ] = len(
        working_ids
    )


    if len(
        working_ids
    ) < 4:

        continue


    gnss_signal_available_history[
        index
    ] = (
        True
    )


    # ========================================================
    # INTEGRITY AVAILABILITY
    # ========================================================

    gnss_integrity_available_history[
        index
    ] = (
        len(
            working_ids
        )
        >=
        minimum_satellites_for_integrity
    )


    # ========================================================
    # PSEUDORANGES
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
    # PHASE 9-E FDIR
    # ========================================================

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
                    previous_gnss_position_guess,

                initial_clock_bias_meters=
                    estimated_clock_bias_meters,

                pseudorange_noise_std=
                    pseudorange_noise_std,

                confidence=
                    gnss_confidence
            )
        )


        gnss_fdir_accepted_history[
            index
        ] = bool(
            fdir_result[
                "measurement_accepted"
            ]
        )

        gnss_fault_detected_history[
            index
        ] = bool(
            fdir_result[
                "fault_detected"
            ]
        )

        gnss_reconfigured_history[
            index
        ] = bool(
            fdir_result[
                "reconfigured"
            ]
        )


        isolated_id = (
            fdir_result[
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

            correct_isolation_history[
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
        # STORE FDIR ACCEPTED SOLUTION
        # ====================================================

        if gnss_fdir_accepted_history[
            index
        ]:

            accepted_solution = (
                fdir_result[
                    "solution"
                ]
            )

            accepted_satellite_ids = [
                str(
                    satellite_id
                ).strip()
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

                gnss_ls_error_history[
                    index
                ] = np.linalg.norm(
                    position_measurement
                    -
                    truth_states[
                        index,
                        0:3
                    ]
                )

                pdop_history[
                    index
                ] = (
                    covariance_result[
                        "pdop"
                    ]
                )


            else:

                gnss_fdir_accepted_history[
                    index
                ] = (
                    False
                )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        gnss_fdir_accepted_history[
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


    position_estimates = np.zeros(
        (
            number_of_epochs,
            3
        )
    )

    position_errors = np.zeros(
        number_of_epochs
    )

    position_sigmas = np.zeros(
        number_of_epochs
    )

    accelerometer_bias_errors = np.zeros(
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

    mode_history = []

    severity_history = np.zeros(
        number_of_epochs,
        dtype=int
    )


    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    position_estimates[
        0
    ] = (
        state[
            0:3
        ]
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

    accelerometer_bias_errors[
        0
    ] = np.linalg.norm(
        state[
            6:9
        ]
        -
        true_accelerometer_bias_body
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

    severity_history[
        0
    ] = (
        initial_decision.severity
    )


    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ====================================================
        # NAVIGATION PREDICTION
        # ====================================================

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


        # ====================================================
        # STAR TRACKER AVAILABILITY FOR MANAGER
        # ====================================================

        star_tracker_available = (
            not star_tracker_outage_mask[
                index
            ]
            and
            not dual_outage_mask[
                index
            ]
        )


        # ====================================================
        # INTEGRITY POLICY
        # ====================================================

        if protected_integrity_policy:

            integrity_flag = (
                gnss_integrity_available_history[
                    index
                ]
            )

        else:

            # Permissive system ignores the structural
            # integrity-availability requirement.
            integrity_flag = (
                True
            )


        # ====================================================
        # RESILIENCE MANAGER
        # ====================================================

        decision = (
            manager.update(
                gnss_signal_available=
                    gnss_signal_available_history[
                        index
                    ],

                gnss_measurement_accepted=
                    gnss_fdir_accepted_history[
                        index
                    ],

                gnss_fault_detected=
                    gnss_fault_detected_history[
                        index
                    ],

                gnss_reconfigured=
                    gnss_reconfigured_history[
                        index
                    ],

                star_tracker_available=
                    star_tracker_available,

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

        severity_history[
            index
        ] = (
            decision.severity
        )


        # ====================================================
        # GNSS MEASUREMENT EXISTS?
        # ====================================================

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


        # ====================================================
        # NAVIGATION UPDATE
        # ====================================================

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


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        position_estimates[
            index
        ] = (
            state[
                0:3
            ]
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

        accelerometer_bias_errors[
            index
        ] = np.linalg.norm(
            state[
                6:9
            ]
            -
            true_accelerometer_bias_body
        )


    return {
        "position_estimates":
            position_estimates,

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "accelerometer_bias_errors":
            accelerometer_bias_errors,

        "navigation_nis":
            navigation_nis,

        "update_used":
            update_used,

        "mode_history":
            mode_history,

        "severity":
            severity_history
    }


# ------------------------------------------------------------
# 14. RUN BOTH ARCHITECTURES
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
# 15. HELPER FUNCTIONS FOR STATISTICS
# ------------------------------------------------------------

def rmse(
    values
):
    return np.sqrt(
        np.mean(
            np.asarray(
                values
            )**2
        )
    )


def compute_window_statistics(
    result,
    mask
):

    errors = (
        result[
            "position_errors"
        ][
            mask
        ]
    )

    sigmas = (
        result[
            "position_sigmas"
        ][
            mask
        ]
    )

    nis = (
        result[
            "navigation_nis"
        ][
            mask
        ]
    )

    valid_nis = np.isfinite(
        nis
    )


    if np.any(
        valid_nis
    ):

        mean_nis = np.mean(
            nis[
                valid_nis
            ]
        )

    else:

        mean_nis = (
            np.nan
        )


    return {
        "rmse":
            rmse(
                errors
            ),

        "max_error":
            np.max(
                errors
            ),

        "mean_sigma":
            np.mean(
                sigmas
            ),

        "mean_nis":
            mean_nis
    }


def compute_global_statistics(
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
            "navigation_nis"
        ]
    )

    valid_nis = np.isfinite(
        nis
    )


    return {
        "rmse":
            rmse(
                errors
            ),

        "max_error":
            np.max(
                errors
            ),

        "final_error":
            errors[
                -1
            ],

        "final_sigma":
            sigmas[
                -1
            ],

        "mean_nis":
            np.mean(
                nis[
                    valid_nis
                ]
            ),

        "updates_used":
            int(
                np.sum(
                    result[
                        "update_used"
                    ]
                )
            )
    }


# ------------------------------------------------------------
# 16. GLOBAL STATISTICS
# ------------------------------------------------------------

permissive_global = (
    compute_global_statistics(
        permissive
    )
)

protected_global = (
    compute_global_statistics(
        protected
    )
)


# ------------------------------------------------------------
# 17. EVENT STATISTICS
# ------------------------------------------------------------

event_masks = {
    "Strong GNSS fault":
        strong_fault_mask,

    "GNSS outage":
        gnss_outage_mask,

    "Star tracker outage":
        star_tracker_outage_mask,

    "Low redundancy fault":
        low_redundancy_fault_mask,

    "Post low redundancy":
        post_low_redundancy_mask,

    "Dual outage":
        dual_outage_mask,

    "Post dual outage":
        post_dual_outage_mask
}


permissive_event_stats = {}

protected_event_stats = {}


for event_name, event_mask in event_masks.items():

    permissive_event_stats[
        event_name
    ] = (
        compute_window_statistics(
            permissive,
            event_mask
        )
    )

    protected_event_stats[
        event_name
    ] = (
        compute_window_statistics(
            protected,
            event_mask
        )
    )


# ------------------------------------------------------------
# 18. FDIR STATISTICS
# ------------------------------------------------------------

strong_fault_epochs = (
    strong_fault_mask
    &
    intentional_fault_history
)

low_redundancy_epochs = (
    low_redundancy_fault_mask
    &
    intentional_fault_history
)


strong_detection_rate = (
    100.0
    *
    np.mean(
        gnss_fault_detected_history[
            strong_fault_epochs
        ]
    )
)

strong_correct_isolation_rate = (
    100.0
    *
    np.mean(
        correct_isolation_history[
            strong_fault_epochs
        ]
    )
)


low_detection_rate = (
    100.0
    *
    np.mean(
        gnss_fault_detected_history[
            low_redundancy_epochs
        ]
    )
)

low_miss_rate = (
    100.0
    -
    low_detection_rate
)


detected_low_mask = (
    low_redundancy_epochs
    &
    gnss_fault_detected_history
)

number_low_detections = np.sum(
    detected_low_mask
)


if number_low_detections > 0:

    low_rejection_given_detection = (
        100.0
        *
        np.mean(
            ~gnss_fdir_accepted_history[
                detected_low_mask
            ]
        )
    )

else:

    low_rejection_given_detection = (
        np.nan
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
    np.mean(
        gnss_fault_detected_history[
            healthy_gnss_mask
        ]
    )
)


# ------------------------------------------------------------
# 19. LOW-REDUNDANCY POLICY METRICS
# ------------------------------------------------------------

missed_low_fault_mask = (
    low_redundancy_epochs
    &
    ~gnss_fault_detected_history
)


permissive_missed_fault_update_rate = (
    100.0
    *
    np.mean(
        permissive[
            "update_used"
        ][
            missed_low_fault_mask
        ]
    )
)


protected_missed_fault_update_rate = (
    100.0
    *
    np.mean(
        protected[
            "update_used"
        ][
            missed_low_fault_mask
        ]
    )
)


# ------------------------------------------------------------
# 20. MODE DURATIONS
#
# Count INTERVALS rather than epochs.
# This avoids the old +dt reporting issue.
# ------------------------------------------------------------

all_modes = [
    ResilienceMode.NOMINAL,
    ResilienceMode.RECOVERY,
    ResilienceMode.GNSS_RECONFIGURED,
    ResilienceMode.GNSS_INTEGRITY_UNAVAILABLE,
    ResilienceMode.GNSS_REJECTED,
    ResilienceMode.GNSS_OUTAGE,
    ResilienceMode.ATTITUDE_DEGRADED,
    ResilienceMode.DUAL_OUTAGE
]


def compute_mode_durations(
    mode_history
):

    durations = {}

    interval_modes = (
        mode_history[
            :-1
        ]
    )

    for mode in all_modes:

        number_intervals = sum(
            current_mode
            ==
            mode
            for current_mode in interval_modes
        )

        durations[
            mode
        ] = (
            number_intervals
            *
            dt
            /
            60.0
        )

    return durations


permissive_mode_durations = (
    compute_mode_durations(
        permissive[
            "mode_history"
        ]
    )
)

protected_mode_durations = (
    compute_mode_durations(
        protected[
            "mode_history"
        ]
    )
)


# ------------------------------------------------------------
# 21. ATTITUDE STATISTICS
# ------------------------------------------------------------

valid_attitude_nis = np.isfinite(
    mekf_nis_history
)

mean_attitude_nis = np.mean(
    mekf_nis_history[
        valid_attitude_nis
    ]
)

attitude_global_rmse_deg = np.rad2deg(
    rmse(
        mekf_attitude_error_history
    )
)

attitude_star_tracker_outage_rmse_deg = np.rad2deg(
    rmse(
        mekf_attitude_error_history[
            star_tracker_outage_mask
        ]
    )
)

attitude_dual_outage_rmse_deg = np.rad2deg(
    rmse(
        mekf_attitude_error_history[
            dual_outage_mask
        ]
    )
)

final_gyro_bias_error_deg_per_second = np.rad2deg(
    mekf_gyro_bias_error_history[
        -1
    ]
)


# ------------------------------------------------------------
# 22. GNSS STATISTICS
# ------------------------------------------------------------

valid_gnss_ls = np.isfinite(
    gnss_ls_error_history
)

gnss_ls_rmse = rmse(
    gnss_ls_error_history[
        valid_gnss_ls
    ]
)

valid_pdop = np.isfinite(
    pdop_history
)

mean_pdop = np.mean(
    pdop_history[
        valid_pdop
    ]
)


# ------------------------------------------------------------
# 23. PERFORMANCE IMPROVEMENT
# ------------------------------------------------------------

low_redundancy_rmse_improvement = (
    permissive_event_stats[
        "Low redundancy fault"
    ][
        "rmse"
    ]
    /
    protected_event_stats[
        "Low redundancy fault"
    ][
        "rmse"
    ]
)

dual_outage_rmse_improvement = (
    permissive_event_stats[
        "Dual outage"
    ][
        "rmse"
    ]
    /
    protected_event_stats[
        "Dual outage"
    ][
        "rmse"
    ]
)

mission_rmse_improvement = (
    permissive_global[
        "rmse"
    ]
    /
    protected_global[
        "rmse"
    ]
)


# ------------------------------------------------------------
# 24. VALIDATION CONDITIONS
# ------------------------------------------------------------

strong_fdir_valid = (
    strong_detection_rate
    >
    95.0
    and
    strong_correct_isolation_rate
    >
    95.0
)

low_redundancy_is_discriminating = (
    low_detection_rate
    <
    90.0
)

protected_blocks_missed_faults = (
    protected_missed_fault_update_rate
    <
    1.0
)

permissive_uses_missed_faults = (
    permissive_missed_fault_update_rate
    >
    90.0
)

protected_low_rmse_better = (
    protected_event_stats[
        "Low redundancy fault"
    ][
        "rmse"
    ]
    <
    permissive_event_stats[
        "Low redundancy fault"
    ][
        "rmse"
    ]
)

protected_global_nis_reasonable = (
    1.5
    <
    protected_global[
        "mean_nis"
    ]
    <
    4.5
)

attitude_consistent = (
    1.5
    <
    mean_attitude_nis
    <
    4.5
)

validation_014a = (
    strong_fdir_valid
    and
    low_redundancy_is_discriminating
    and
    protected_blocks_missed_faults
    and
    permissive_uses_missed_faults
    and
    protected_low_rmse_better
    and
    protected_global_nis_reasonable
    and
    attitude_consistent
)


# ------------------------------------------------------------
# 25. PRINT HEADER
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================================================================"
)

print(
    "AURORA — Experience 014-A"
)

print(
    "Full mission resilience : permissive vs AURORA protected"
)

print(
    "============================================================================================================================================"
)

print(
    f"Duree mission : "
    f"{simulation_duration_minutes:.1f} min"
)

print(
    f"Pas temporel : "
    f"{dt:.1f} s"
)

print(
    f"Minimum satellites pour integrite : "
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


# ------------------------------------------------------------
# 26. PRINT FDIR
# ------------------------------------------------------------

print(
    "----- FDIR GNSS -----"
)

print(
    f"Detection faute forte 20-30 min : "
    f"{strong_detection_rate:.2f} %"
)

print(
    f"Isolation correcte faute forte : "
    f"{strong_correct_isolation_rate:.2f} %"
)

print(
    f"Detection faible redondance 105-115 min : "
    f"{low_detection_rate:.2f} %"
)

print(
    f"Non-detection faible redondance : "
    f"{low_miss_rate:.2f} %"
)

print(
    f"Rejet conditionnel si detection : "
    f"{low_rejection_given_detection:.2f} %"
)

print(
    f"Taux fausse alarme : "
    f"{false_alarm_rate:.2f} %"
)

print()


# ------------------------------------------------------------
# 27. PRINT ATTITUDE
# ------------------------------------------------------------

print(
    "----- ATTITUDE MEKF -----"
)

print(
    f"RMSE attitude globale : "
    f"{attitude_global_rmse_deg:.4f} deg"
)

print(
    f"RMSE attitude perte ST : "
    f"{attitude_star_tracker_outage_rmse_deg:.4f} deg"
)

print(
    f"RMSE attitude dual outage : "
    f"{attitude_dual_outage_rmse_deg:.4f} deg"
)

print(
    f"Erreur biais gyro finale : "
    f"{final_gyro_bias_error_deg_per_second:.6f} deg/s"
)

print(
    f"NIS attitude moyen : "
    f"{mean_attitude_nis:.3f}"
)

print(
    "Valeur theorique NIS attitude ~3"
)

print()


# ------------------------------------------------------------
# 28. PRINT GLOBAL NAVIGATION
# ------------------------------------------------------------

print(
    "----- PERFORMANCE MISSION GLOBALE -----"
)

header_global = (
    f"{'Architecture':>20} | "
    f"{'RMSE [m]':>10} | "
    f"{'Max [m]':>10} | "
    f"{'Err finale':>11} | "
    f"{'Sigma finale':>12} | "
    f"{'NIS moyen':>10} | "
    f"{'Updates':>8}"
)

print(
    header_global
)

print(
    "-" * len(
        header_global
    )
)

print(
    f"{'Permissive':>20} | "
    f"{permissive_global['rmse']:10.3f} | "
    f"{permissive_global['max_error']:10.3f} | "
    f"{permissive_global['final_error']:11.3f} | "
    f"{permissive_global['final_sigma']:12.3f} | "
    f"{permissive_global['mean_nis']:10.3f} | "
    f"{permissive_global['updates_used']:8d}"
)

print(
    f"{'AURORA protected':>20} | "
    f"{protected_global['rmse']:10.3f} | "
    f"{protected_global['max_error']:10.3f} | "
    f"{protected_global['final_error']:11.3f} | "
    f"{protected_global['final_sigma']:12.3f} | "
    f"{protected_global['mean_nis']:10.3f} | "
    f"{protected_global['updates_used']:8d}"
)

print()


# ------------------------------------------------------------
# 29. PRINT EVENT TABLE
# ------------------------------------------------------------

print(
    "----- PERFORMANCE PAR EVENEMENT -----"
)

header_event = (
    f"{'Evenement':>24} | "
    f"{'RMSE permissive':>15} | "
    f"{'RMSE protected':>14} | "
    f"{'Max permissive':>15} | "
    f"{'Max protected':>13}"
)

print(
    header_event
)

print(
    "-" * len(
        header_event
    )
)

for event_name in event_masks.keys():

    permissive_stats = (
        permissive_event_stats[
            event_name
        ]
    )

    protected_stats = (
        protected_event_stats[
            event_name
        ]
    )

    print(
        f"{event_name:>24} | "
        f"{permissive_stats['rmse']:15.3f} | "
        f"{protected_stats['rmse']:14.3f} | "
        f"{permissive_stats['max_error']:15.3f} | "
        f"{protected_stats['max_error']:13.3f}"
    )

print()


# ------------------------------------------------------------
# 30. PRINT INTEGRITY POLICY
# ------------------------------------------------------------

print(
    "----- INTEGRITY POLICY -----"
)

print(
    f"Updates sur fautes non detectees — permissive : "
    f"{permissive_missed_fault_update_rate:.2f} %"
)

print(
    f"Updates sur fautes non detectees — protected : "
    f"{protected_missed_fault_update_rate:.2f} %"
)

print(
    f"Facteur amelioration RMSE faible redondance : "
    f"{low_redundancy_rmse_improvement:.2f}"
)

print(
    f"Facteur amelioration RMSE dual outage : "
    f"{dual_outage_rmse_improvement:.2f}"
)

print(
    f"Facteur amelioration RMSE mission globale : "
    f"{mission_rmse_improvement:.2f}"
)

print()


# ------------------------------------------------------------
# 31. PRINT MODE DURATIONS
# ------------------------------------------------------------

print(
    "----- TEMPS DANS LES MODES [min] -----"
)

header_mode = (
    f"{'Mode':>28} | "
    f"{'Permissive':>12} | "
    f"{'Protected':>12}"
)

print(
    header_mode
)

print(
    "-" * len(
        header_mode
    )
)

for mode in all_modes:

    print(
        f"{mode.value:>28} | "
        f"{permissive_mode_durations[mode]:12.2f} | "
        f"{protected_mode_durations[mode]:12.2f}"
    )

print()


# ------------------------------------------------------------
# 32. PRINT VALIDATION
# ------------------------------------------------------------

print(
    "----- VALIDATION -----"
)

print(
    f"FDIR forte redondance valide : "
    f"{strong_fdir_valid}"
)

print(
    f"Scenario faible redondance discriminant : "
    f"{low_redundancy_is_discriminating}"
)

print(
    f"Protected bloque les fautes manquees : "
    f"{protected_blocks_missed_faults}"
)

print(
    f"Permissive utilise les fautes manquees : "
    f"{permissive_uses_missed_faults}"
)

print(
    f"RMSE faible redondance protegee meilleure : "
    f"{protected_low_rmse_better}"
)

print(
    f"NIS navigation protected coherent : "
    f"{protected_global_nis_reasonable}"
)

print(
    f"MEKF coherent : "
    f"{attitude_consistent}"
)

print()

print(
    f"VALIDATION GLOBALE 014-A : "
    f"{validation_014a}"
)

print(
    "============================================================================================================================================"
)


# ------------------------------------------------------------
# 33. FIGURE — POSITION ERROR
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 7)
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
    20.0,
    30.0,
    alpha=0.10,
    label="GNSS fault"
)

plt.axvspan(
    40.0,
    55.0,
    alpha=0.10,
    label="GNSS outage"
)

plt.axvspan(
    65.0,
    75.0,
    alpha=0.10,
    label="ST outage"
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
    "AURORA — Full mission navigation resilience"
)

plt.grid(
    True
)

plt.legend(
    ncol=3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 34. FIGURE — ERROR VS COVARIANCE
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 7)
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
    alpha=0.10
)

plt.axvspan(
    125.0,
    140.0,
    alpha=0.10
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Position [m]"
)

plt.title(
    "AURORA — Navigation error vs covariance"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 35. FIGURE — ATTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 6)
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_error_history
    ),
    label="Erreur attitude"
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_sigma_history
    ),
    linestyle="--",
    label="Sigma attitude 3D"
)

plt.axvspan(
    65.0,
    75.0,
    alpha=0.10,
    label="ST outage"
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
    "Attitude [deg]"
)

plt.title(
    "AURORA — MEKF attitude over full mission"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 36. FIGURE — GNSS UPDATE DECISIONS
# ------------------------------------------------------------

plt.figure(
    figsize=(15, 5)
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
    label="AURORA protected"
)

plt.axvspan(
    105.0,
    115.0,
    alpha=0.10,
    label="5 sats + fault"
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
    "AURORA — Mission GNSS update decisions"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 37. FIGURE — PROTECTED MODE HISTORY
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
    figsize=(15, 7)
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

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Mode"
)

plt.title(
    "AURORA — Protected resilience modes over full mission"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()