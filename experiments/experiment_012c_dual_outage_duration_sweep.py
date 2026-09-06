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
    solve_gnss_least_squares_transmit_time,
    compute_transmit_time_solution_covariance
)


# ============================================================
# AURORA
# Experience 012-C
#
# Sweep de duree du double outage :
#
#     GNSS indisponible
#          +
#     star tracker indisponible
#
#
# Phase d'apprentissage :
#
#     0 -> 60 min
#
# GNSS + star tracker disponibles.
#
#
# A t = 60 min :
#
# on part volontairement d'un etat corrige
# par GNSS et star tracker.
#
#
# Puis on teste :
#
#     5 min
#     10 min
#     20 min
#     40 min
#     60 min
#
# sans GNSS ni star tracker.
#
#
# Comparaison navigation :
#
#     - attitude parfaite
#     - attitude MEKF propagee gyro-only
#
#
# Objectif :
#
# determiner a partir de quelle duree
# l'erreur d'attitude devient significative
# pour la navigation translationnelle.
# ============================================================


# ------------------------------------------------------------
# 1. DUREES DE BLACKOUT
# ------------------------------------------------------------

learning_duration_minutes = (
    60.0
)


outage_durations_minutes = np.array([
    5.0,
    10.0,
    20.0,
    40.0,
    60.0
])


maximum_outage_duration_minutes = np.max(
    outage_durations_minutes
)


simulation_duration_minutes = (
    learning_duration_minutes
    +
    maximum_outage_duration_minutes
)


simulation_duration = (
    simulation_duration_minutes
    * 60.0
)


# ------------------------------------------------------------
# 2. TEMPS
# ------------------------------------------------------------

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
    - time[0]
)


time_minutes = (
    time
    / 60.0
)


learning_end_index = int(
    round(
        learning_duration_minutes
        * 60.0
        / dt
    )
)


# ------------------------------------------------------------
# 3. ORBITE
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


# ------------------------------------------------------------
# 4. VERITE ORBITALE
# ------------------------------------------------------------

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
        - time
    )
)


# ------------------------------------------------------------
# 5. ATTITUDE VRAIE
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
# 6. VITESSE ANGULAIRE VRAIE BODY
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
        / dt
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
# 8. STAR TRACKER
# ------------------------------------------------------------

star_tracker_period_seconds = (
    60.0
)


star_tracker_interval_epochs = int(
    round(
        star_tracker_period_seconds
        / dt
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


star_tracker_available = np.zeros(
    number_of_epochs,
    dtype=bool
)


star_tracker_measurements = np.full(
    (
        number_of_epochs,
        4
    ),
    np.nan
)


for index in range(
    star_tracker_interval_epochs,
    learning_end_index + 1,
    star_tracker_interval_epochs
):

    result = (
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


    star_tracker_measurements[
        index
    ] = (
        result[
            "measurement"
        ]
    )


    star_tracker_available[
        index
    ] = (
        True
    )


# ------------------------------------------------------------
# 9. PARAMETRES MEKF
# ------------------------------------------------------------

initial_attitude_sigma_deg = (
    0.10
)


initial_attitude_sigma = np.deg2rad(
    initial_attitude_sigma_deg
)


initial_gyro_bias_sigma_deg_per_second = (
    0.005
)


initial_gyro_bias_sigma = np.deg2rad(
    initial_gyro_bias_sigma_deg_per_second
)


mekf_covariance = np.diag([
    initial_attitude_sigma**2,
    initial_attitude_sigma**2,
    initial_attitude_sigma**2,

    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2,
    initial_gyro_bias_sigma**2
])


gyro_bias_random_walk_density_deg = (
    1.0e-7
)


gyro_bias_random_walk_density = np.deg2rad(
    gyro_bias_random_walk_density_deg
)


# ------------------------------------------------------------
# 10. APPRENTISSAGE MEKF JUSQU'A 60 MIN
# ------------------------------------------------------------

mekf_quaternion = (
    true_quaternion_history[
        0
    ].copy()
)


mekf_gyro_bias = np.zeros(
    3
)


pre_mekf_rotation_history = np.zeros(
    (
        learning_end_index + 1,
        3,
        3
    )
)


pre_mekf_rotation_history[
    0
] = quaternion_to_rotation_matrix(
    mekf_quaternion
)


pre_mekf_nis = []


for index in range(
    learning_end_index
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
                    index
                ],

            dt=
                dt,

            gyro_noise_std=
                gyro_noise_std,

            gyro_bias_random_walk_density=
                gyro_bias_random_walk_density
        )
    )


    predicted_quaternion = (
        prediction[
            "quaternion"
        ]
    )


    predicted_bias = (
        prediction[
            "gyro_bias"
        ]
    )


    predicted_covariance = (
        prediction[
            "covariance"
        ]
    )


    if star_tracker_available[
        index + 1
    ]:

        update = (
            update_attitude_mekf_with_star_tracker(
                predicted_quaternion=
                    predicted_quaternion,

                predicted_gyro_bias=
                    predicted_bias,

                predicted_covariance=
                    predicted_covariance,

                star_tracker_quaternion=
                    star_tracker_measurements[
                        index + 1
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


        pre_mekf_nis.append(
            update[
                "nis"
            ]
        )


    else:

        mekf_quaternion = (
            predicted_quaternion
        )


        mekf_gyro_bias = (
            predicted_bias
        )


        mekf_covariance = (
            predicted_covariance
        )


    pre_mekf_rotation_history[
        index + 1
    ] = quaternion_to_rotation_matrix(
        mekf_quaternion
    )


# ------------------------------------------------------------
# 11. ETAT MEKF AU DEBUT DU BLACKOUT
# ------------------------------------------------------------

blackout_initial_quaternion = (
    mekf_quaternion.copy()
)


blackout_initial_gyro_bias = (
    mekf_gyro_bias.copy()
)


blackout_initial_mekf_covariance = (
    mekf_covariance.copy()
)


initial_blackout_attitude_error = (
    quaternion_attitude_error_angle(
        reference_quaternion=
            true_quaternion_history[
                learning_end_index
            ],

        estimated_quaternion=
            blackout_initial_quaternion
    )
)


initial_blackout_attitude_sigma = np.sqrt(
    np.trace(
        blackout_initial_mekf_covariance[
            0:3,
            0:3
        ]
    )
)


initial_blackout_gyro_bias_error = np.linalg.norm(
    blackout_initial_gyro_bias
    -
    true_gyro_bias_body
)


# ------------------------------------------------------------
# 12. FORCE SPECIFIQUE + ACCELEROMETRE
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


true_specific_force_body_history = np.zeros(
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


    true_specific_force_body_history[
        index
    ] = (
        result[
            "true_specific_force_body"
        ]
    )


# ------------------------------------------------------------
# 13. GNSS POUR LA PHASE D'APPRENTISSAGE
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


gnss_rng = np.random.default_rng(
    2026
)


gnss_positions = np.full(
    (
        learning_end_index + 1,
        3
    ),
    np.nan
)


gnss_covariances = np.full(
    (
        learning_end_index + 1,
        3,
        3
    ),
    np.nan
)


gnss_solution_available = np.zeros(
    learning_end_index + 1,
    dtype=bool
)


gnss_ls_errors = np.full(
    learning_end_index + 1,
    np.nan
)


pdop_values = np.full(
    learning_end_index + 1,
    np.nan
)


estimated_clock_bias_meters = (
    0.0
)


for index in range(
    1,
    learning_end_index + 1
):

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


    if len(
        visible_ids
    ) < 4:

        continue


    pseudorange_result = (
        simulate_pseudoranges_with_transmit_time(
            receiver_position=
                truth_states[
                    index,
                    0:3
                ],

            satellite_ids=
                visible_ids,

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


    try:

        solution = (
            solve_gnss_least_squares_transmit_time(
                satellite_ids=
                    visible_ids,

                pseudoranges=
                    pseudorange_result[
                        "pseudoranges"
                    ],

                reception_time_seconds=
                    time[
                        index
                    ],

                initial_position=
                    truth_states[
                        index,
                        0:3
                    ]
                    +
                    np.array([
                        1000.0,
                        -800.0,
                        600.0
                    ]),

                initial_clock_bias_meters=
                    estimated_clock_bias_meters
            )
        )


        if not solution[
            "converged"
        ]:

            continue


        estimated_clock_bias_meters = (
            solution[
                "clock_bias_meters"
            ]
        )


        gnss_position = (
            solution[
                "position"
            ]
        )


        covariance_result = (
            compute_transmit_time_solution_covariance(
                receiver_position=
                    gnss_position,

                satellite_ids=
                    visible_ids,

                reception_time_seconds=
                    time[
                        index
                    ],

                pseudorange_noise_std=
                    pseudorange_noise_std
            )
        )


        gnss_positions[
            index
        ] = (
            gnss_position
        )


        gnss_covariances[
            index
        ] = (
            covariance_result[
                "position_covariance"
            ]
        )


        gnss_ls_errors[
            index
        ] = np.linalg.norm(
            gnss_position
            -
            truth_states[
                index,
                0:3
            ]
        )


        pdop_values[
            index
        ] = (
            covariance_result[
                "pdop"
            ]
        )


        gnss_solution_available[
            index
        ] = (
            True
        )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        continue


# ------------------------------------------------------------
# 14. INITIALISATION NAVIGATION
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


initial_orbital_estimate = (
    true_initial_orbital_state
    +
    np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )
)


navigation_state = np.concatenate(
    (
        initial_orbital_estimate,
        np.zeros(
            3
        )
    )
)


initial_position_sigma = (
    100.0
)


initial_velocity_sigma = (
    0.10
)


initial_accelerometer_bias_sigma = (
    5.0e-6
)


navigation_covariance = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2,

    initial_accelerometer_bias_sigma**2,
    initial_accelerometer_bias_sigma**2,
    initial_accelerometer_bias_sigma**2
])


accelerometer_bias_random_walk_density = (
    1.0e-10
)


pre_navigation_nis = []


# ------------------------------------------------------------
# 15. APPRENTISSAGE NAVIGATION JUSQU'A 60 MIN
#
# Important :
#
# On utilise ici l'attitude MEKF reelle.
#
# Les deux branches futures partiront donc
# EXACTEMENT du meme etat navigation.
# ------------------------------------------------------------

for index in range(
    1,
    learning_end_index + 1
):

    (
        predicted_state,
        predicted_covariance
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
                pre_mekf_rotation_history[
                    index - 1
                ],

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                accelerometer_bias_random_walk_density
        )
    )


    if gnss_solution_available[
        index
    ]:

        update_result = (
            ekf_update_position_augmented(
                predicted_state=
                    predicted_state,

                predicted_covariance=
                    predicted_covariance,

                position_measurement=
                    gnss_positions[
                        index
                    ],

                measurement_noise_covariance=
                    gnss_covariances[
                        index
                    ]
            )
        )


        innovation = (
            update_result[
                "innovation"
            ]
        )


        S = (
            update_result[
                "innovation_covariance"
            ]
        )


        pre_navigation_nis.append(
            innovation.T
            @ np.linalg.solve(
                S,
                innovation
            )
        )


        navigation_state = (
            update_result[
                "state"
            ]
        )


        navigation_covariance = (
            update_result[
                "covariance"
            ]
        )


    else:

        navigation_state = (
            predicted_state
        )


        navigation_covariance = (
            predicted_covariance
        )


# ------------------------------------------------------------
# 16. ETAT NAVIGATION AU DEBUT DU BLACKOUT
# ------------------------------------------------------------

blackout_initial_navigation_state = (
    navigation_state.copy()
)


blackout_initial_navigation_covariance = (
    navigation_covariance.copy()
)


initial_navigation_position_error = np.linalg.norm(
    blackout_initial_navigation_state[
        0:3
    ]
    -
    truth_states[
        learning_end_index,
        0:3
    ]
)


initial_navigation_bias_error = np.linalg.norm(
    blackout_initial_navigation_state[
        6:9
    ]
    -
    true_accelerometer_bias_body
)


# ------------------------------------------------------------
# 17. PROPAGATION ATTITUDE SANS STAR TRACKER
# ------------------------------------------------------------

def propagate_attitude_blackout(
    duration_minutes
):

    number_of_steps = int(
        round(
            duration_minutes
            * 60.0
            / dt
        )
    )


    quaternion = (
        blackout_initial_quaternion.copy()
    )


    gyro_bias = (
        blackout_initial_gyro_bias.copy()
    )


    covariance = (
        blackout_initial_mekf_covariance.copy()
    )


    rotation_history = np.zeros(
        (
            number_of_steps + 1,
            3,
            3
        )
    )


    attitude_errors = np.zeros(
        number_of_steps + 1
    )


    attitude_sigmas = np.zeros(
        number_of_steps + 1
    )


    gyro_bias_errors = np.zeros(
        number_of_steps + 1
    )


    acceleration_projection_errors = np.zeros(
        number_of_steps
    )


    rotation_history[
        0
    ] = quaternion_to_rotation_matrix(
        quaternion
    )


    attitude_errors[
        0
    ] = (
        initial_blackout_attitude_error
    )


    attitude_sigmas[
        0
    ] = (
        initial_blackout_attitude_sigma
    )


    gyro_bias_errors[
        0
    ] = (
        initial_blackout_gyro_bias_error
    )


    for step in range(
        number_of_steps
    ):

        global_index = (
            learning_end_index
            +
            step
        )


        estimated_rotation = (
            rotation_history[
                step
            ]
        )


        true_rotation = (
            true_rotation_history[
                global_index
            ]
        )


        true_force_body = (
            true_specific_force_body_history[
                global_index
            ]
        )


        acceleration_projection_errors[
            step
        ] = np.linalg.norm(
            estimated_rotation
            @ true_force_body
            -
            true_rotation
            @ true_force_body
        )


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
                        global_index
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


        rotation_history[
            step + 1
        ] = quaternion_to_rotation_matrix(
            quaternion
        )


        attitude_errors[
            step + 1
        ] = (
            quaternion_attitude_error_angle(
                reference_quaternion=
                    true_quaternion_history[
                        global_index + 1
                    ],

                estimated_quaternion=
                    quaternion
            )
        )


        attitude_sigmas[
            step + 1
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        gyro_bias_errors[
            step + 1
        ] = np.linalg.norm(
            gyro_bias
            -
            true_gyro_bias_body
        )


    return {
        "rotation_history":
            rotation_history,

        "attitude_errors":
            attitude_errors,

        "attitude_sigmas":
            attitude_sigmas,

        "gyro_bias_errors":
            gyro_bias_errors,

        "acceleration_projection_errors":
            acceleration_projection_errors,

        "number_of_steps":
            number_of_steps
    }


# ------------------------------------------------------------
# 18. PROPAGATION NAVIGATION SANS GNSS
# ------------------------------------------------------------

def propagate_navigation_blackout(
    duration_minutes,
    rotation_history
):

    number_of_steps = int(
        round(
            duration_minutes
            * 60.0
            / dt
        )
    )


    state = (
        blackout_initial_navigation_state.copy()
    )


    covariance = (
        blackout_initial_navigation_covariance.copy()
    )


    position_estimates = np.zeros(
        (
            number_of_steps + 1,
            3
        )
    )


    position_errors = np.zeros(
        number_of_steps + 1
    )


    position_sigmas = np.zeros(
        number_of_steps + 1
    )


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
            learning_end_index,
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


    for step in range(
        number_of_steps
    ):

        global_index = (
            learning_end_index
            +
            step
        )


        (
            state,
            covariance
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
                        global_index
                    ],

                rotation_body_to_eci=
                    rotation_history[
                        step
                    ],

                accelerometer_noise_std=
                    accelerometer_noise_std,

                bias_random_walk_density=
                    accelerometer_bias_random_walk_density
            )
        )


        position_estimates[
            step + 1
        ] = (
            state[
                0:3
            ]
        )


        position_errors[
            step + 1
        ] = np.linalg.norm(
            state[
                0:3
            ]
            -
            truth_states[
                global_index + 1,
                0:3
            ]
        )


        position_sigmas[
            step + 1
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


    return {
        "position_estimates":
            position_estimates,

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas
    }


# ------------------------------------------------------------
# 19. STOCKAGE SWEEP
# ------------------------------------------------------------

number_of_cases = len(
    outage_durations_minutes
)


final_attitude_error_deg = np.zeros(
    number_of_cases
)


final_attitude_sigma_deg = np.zeros(
    number_of_cases
)


attitude_rms_deg = np.zeros(
    number_of_cases
)


final_gyro_bias_error_deg_per_second = np.zeros(
    number_of_cases
)


mean_projection_acceleration_error = np.zeros(
    number_of_cases
)


perfect_navigation_rmse = np.zeros(
    number_of_cases
)


mekf_navigation_rmse = np.zeros(
    number_of_cases
)


perfect_navigation_final_error = np.zeros(
    number_of_cases
)


mekf_navigation_final_error = np.zeros(
    number_of_cases
)


mekf_navigation_final_sigma = np.zeros(
    number_of_cases
)


position_separation_rms = np.zeros(
    number_of_cases
)


position_separation_final = np.zeros(
    number_of_cases
)


# ------------------------------------------------------------
# 20. SWEEP
# ------------------------------------------------------------

for (
    case_index,
    duration_minutes
) in enumerate(
    outage_durations_minutes
):

    attitude_result = (
        propagate_attitude_blackout(
            duration_minutes=
                duration_minutes
        )
    )


    number_of_steps = (
        attitude_result[
            "number_of_steps"
        ]
    )


    # --------------------------------------------------------
    # Navigation avec attitude parfaite
    # --------------------------------------------------------

    perfect_rotation_branch = (
        true_rotation_history[
            learning_end_index:
            learning_end_index
            +
            number_of_steps
            +
            1
        ]
    )


    perfect_navigation = (
        propagate_navigation_blackout(
            duration_minutes=
                duration_minutes,

            rotation_history=
                perfect_rotation_branch
        )
    )


    # --------------------------------------------------------
    # Navigation avec attitude MEKF sans ST
    # --------------------------------------------------------

    mekf_navigation = (
        propagate_navigation_blackout(
            duration_minutes=
                duration_minutes,

            rotation_history=
                attitude_result[
                    "rotation_history"
                ]
        )
    )


    # --------------------------------------------------------
    # Attitude statistics
    # --------------------------------------------------------

    final_attitude_error_deg[
        case_index
    ] = np.rad2deg(
        attitude_result[
            "attitude_errors"
        ][
            -1
        ]
    )


    final_attitude_sigma_deg[
        case_index
    ] = np.rad2deg(
        attitude_result[
            "attitude_sigmas"
        ][
            -1
        ]
    )


    attitude_rms_deg[
        case_index
    ] = np.rad2deg(
        np.sqrt(
            np.mean(
                attitude_result[
                    "attitude_errors"
                ]**2
            )
        )
    )


    final_gyro_bias_error_deg_per_second[
        case_index
    ] = np.rad2deg(
        attitude_result[
            "gyro_bias_errors"
        ][
            -1
        ]
    )


    mean_projection_acceleration_error[
        case_index
    ] = np.mean(
        attitude_result[
            "acceleration_projection_errors"
        ]
    )


    # --------------------------------------------------------
    # Navigation statistics
    #
    # Exclusion de l'epoque initiale commune.
    # --------------------------------------------------------

    perfect_errors = (
        perfect_navigation[
            "position_errors"
        ][
            1:
        ]
    )


    mekf_errors = (
        mekf_navigation[
            "position_errors"
        ][
            1:
        ]
    )


    perfect_navigation_rmse[
        case_index
    ] = np.sqrt(
        np.mean(
            perfect_errors**2
        )
    )


    mekf_navigation_rmse[
        case_index
    ] = np.sqrt(
        np.mean(
            mekf_errors**2
        )
    )


    perfect_navigation_final_error[
        case_index
    ] = (
        perfect_navigation[
            "position_errors"
        ][
            -1
        ]
    )


    mekf_navigation_final_error[
        case_index
    ] = (
        mekf_navigation[
            "position_errors"
        ][
            -1
        ]
    )


    mekf_navigation_final_sigma[
        case_index
    ] = (
        mekf_navigation[
            "position_sigmas"
        ][
            -1
        ]
    )


    separation_history = np.linalg.norm(
        mekf_navigation[
            "position_estimates"
        ]
        -
        perfect_navigation[
            "position_estimates"
        ],
        axis=1
    )


    position_separation_rms[
        case_index
    ] = np.sqrt(
        np.mean(
            separation_history[
                1:
            ]**2
        )
    )


    position_separation_final[
        case_index
    ] = (
        separation_history[
            -1
        ]
    )


# ------------------------------------------------------------
# 21. STATISTIQUES PRE-BLACKOUT
# ------------------------------------------------------------

valid_gnss = np.isfinite(
    gnss_ls_errors
)


valid_pdop = np.isfinite(
    pdop_values
)


gnss_ls_rmse = np.sqrt(
    np.mean(
        gnss_ls_errors[
            valid_gnss
        ]**2
    )
)


mean_pdop = np.mean(
    pdop_values[
        valid_pdop
    ]
)


mean_pre_mekf_nis = np.mean(
    pre_mekf_nis
)


mean_pre_navigation_nis = np.mean(
    pre_navigation_nis
)


# ------------------------------------------------------------
# 22. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "=================================================================================================================================================="
)


print(
    "AURORA — Experience 012-C"
)


print(
    "Sweep de duree du double outage GNSS + star tracker"
)


print(
    "=================================================================================================================================================="
)


print(
    f"Phase apprentissage : "
    f"{learning_duration_minutes:.1f} min"
)


print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- ETAT AU DEBUT DU BLACKOUT -----"
)


print(
    f"Erreur attitude : "
    f"{np.rad2deg(initial_blackout_attitude_error):.4f} deg"
)


print(
    f"Sigma attitude 3D : "
    f"{np.rad2deg(initial_blackout_attitude_sigma):.4f} deg"
)


print(
    f"Erreur biais gyro : "
    f"{np.rad2deg(initial_blackout_gyro_bias_error):.6f} deg/s"
)


print(
    f"Erreur position navigation : "
    f"{initial_navigation_position_error:.3f} m"
)


print(
    f"Erreur biais accelerometre : "
    f"{initial_navigation_bias_error:.6e} m/s^2"
)


print()


print(
    "----- VALIDATION PHASE D'APPRENTISSAGE -----"
)


print(
    f"RMSE GNSS LS : "
    f"{gnss_ls_rmse:.3f} m"
)


print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)


print(
    f"NIS MEKF moyen : "
    f"{mean_pre_mekf_nis:.3f}"
)


print(
    f"NIS navigation moyen : "
    f"{mean_pre_navigation_nis:.3f}"
)


print()


print(
    "----- SWEEP DOUBLE OUTAGE -----"
)


header = (
    f"{'Duree':>7} | "
    f"{'Att fin':>9} | "
    f"{'Sigma att':>9} | "
    f"{'da proj moy':>12} | "
    f"{'RMSE perfect':>12} | "
    f"{'RMSE MEKF':>10} | "
    f"{'Sep RMS':>9} | "
    f"{'Sep fin':>9} | "
    f"{'Err fin MEKF':>12} | "
    f"{'Sigma nav':>10}"
)


print(
    header
)


print(
    "-" * len(
        header
    )
)


for case_index in range(
    number_of_cases
):

    print(
        f"{outage_durations_minutes[case_index]:6.1f}m | "
        f"{final_attitude_error_deg[case_index]:8.4f}° | "
        f"{final_attitude_sigma_deg[case_index]:8.4f}° | "
        f"{mean_projection_acceleration_error[case_index]:12.3e} | "
        f"{perfect_navigation_rmse[case_index]:12.3f} | "
        f"{mekf_navigation_rmse[case_index]:10.3f} | "
        f"{position_separation_rms[case_index]:9.3f} | "
        f"{position_separation_final[case_index]:9.3f} | "
        f"{mekf_navigation_final_error[case_index]:12.3f} | "
        f"{mekf_navigation_final_sigma[case_index]:10.3f}"
    )


print(
    "=================================================================================================================================================="
)


# ------------------------------------------------------------
# 23. FIGURE ATTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    outage_durations_minutes,
    final_attitude_error_deg,
    marker="o",
    label="Erreur attitude finale"
)


plt.plot(
    outage_durations_minutes,
    final_attitude_sigma_deg,
    marker="o",
    linestyle="--",
    label="Sigma attitude 3D"
)


plt.xlabel(
    "Duree double outage [min]"
)


plt.ylabel(
    "Attitude [deg]"
)


plt.title(
    "AURORA — Croissance de l'incertitude attitude sans star tracker"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 24. FIGURE SEPARATION NAVIGATION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    outage_durations_minutes,
    position_separation_final,
    marker="o"
)


plt.xlabel(
    "Duree double outage [min]"
)


plt.ylabel(
    "Separation position finale [m]"
)


plt.title(
    "AURORA — Effet pur de l'attitude estimee sur la position"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 25. FIGURE RMSE NAVIGATION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    outage_durations_minutes,
    perfect_navigation_rmse,
    marker="o",
    label="Attitude parfaite"
)


plt.plot(
    outage_durations_minutes,
    mekf_navigation_rmse,
    marker="o",
    label="Attitude MEKF sans star tracker"
)


plt.xlabel(
    "Duree double outage [min]"
)


plt.ylabel(
    "RMSE position [m]"
)


plt.title(
    "AURORA — Navigation vs duree de perte GNSS + star tracker"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 26. FIGURE PROJECTION ACCELERATION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.plot(
    outage_durations_minutes,
    mean_projection_acceleration_error,
    marker="o"
)


plt.xlabel(
    "Duree double outage [min]"
)


plt.ylabel(
    "Erreur acceleration moyenne [m/s²]"
)


plt.title(
    "AURORA — Erreur de projection accelerometre due a l'attitude"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()