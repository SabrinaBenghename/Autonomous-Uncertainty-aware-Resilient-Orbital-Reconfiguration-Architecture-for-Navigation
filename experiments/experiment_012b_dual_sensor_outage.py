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
# Experience 012-B
#
# Perte simultanee :
#
#     GNSS
#       +
#     star tracker
#
# entre 60 et 80 min.
#
#
# Comparaison attitude :
#
#   1. MEKF avec star tracker toujours disponible
#   2. MEKF avec blackout star tracker pendant outage GNSS
#
#
# Comparaison navigation :
#
#   1. attitude parfaite
#   2. attitude MEKF avec ST disponible
#   3. attitude MEKF avec ST indisponible
#
#
# Pendant le double outage :
#
#   Navigation :
#       dynamique orbitale
#       + accelerometre
#
#   Attitude :
#       gyro
#       + biais gyro deja estime
#
#
# Objectif :
#
# tester la resilience couplee attitude/navigation
# lorsque les deux sources absolues :
#
#     GNSS
#     star tracker
#
# sont indisponibles simultanement.
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
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
# 2. TEMPS
# ------------------------------------------------------------

simulation_duration_minutes = (
    100.0
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
    - time[0]
)

time_minutes = (
    time
    / 60.0
)


# ------------------------------------------------------------
# 3. DOUBLE OUTAGE
# ------------------------------------------------------------

outage_start_minutes = (
    60.0
)

outage_end_minutes = (
    80.0
)

outage_mask = (
    (
        time_minutes
        >= outage_start_minutes
    )
    &
    (
        time_minutes
        <= outage_end_minutes
    )
)

gnss_allowed = (
    ~outage_mask
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
#
# Les memes mesures potentielles sont generees
# pour les deux architectures.
#
# Le cas dual-outage les ignore entre 60 et 80 min.
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

star_tracker_measurement_available = np.zeros(
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
    number_of_epochs,
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

    star_tracker_measurement_available[
        index
    ] = (
        True
    )


# ------------------------------------------------------------
# 9. DEUX MASQUES STAR TRACKER
# ------------------------------------------------------------

star_tracker_continuous_mask = (
    star_tracker_measurement_available.copy()
)

star_tracker_dual_outage_mask = (
    star_tracker_measurement_available
    &
    ~outage_mask
)


# ------------------------------------------------------------
# 10. PARAMETRES MEKF
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

initial_mekf_covariance = np.diag([
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
# 11. FONCTION GENERIQUE MEKF
# ------------------------------------------------------------

def run_mekf_case(
    star_tracker_use_mask
):
    """
    Execute le meme MEKF avec un masque
    de disponibilite star tracker donne.
    """

    quaternion = (
        true_quaternion_history[
            0
        ].copy()
    )

    gyro_bias = np.zeros(
        3
    )

    covariance = (
        initial_mekf_covariance.copy()
    )


    quaternion_history = np.zeros(
        (
            number_of_epochs,
            4
        )
    )

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

    bias_error_history = np.zeros(
        number_of_epochs
    )

    bias_estimate_history = np.zeros(
        (
            number_of_epochs,
            3
        )
    )

    nis_history = np.full(
        number_of_epochs,
        np.nan
    )


    quaternion_history[
        0
    ] = quaternion

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

    bias_error_history[
        0
    ] = np.linalg.norm(
        gyro_bias
        -
        true_gyro_bias_body
    )

    bias_estimate_history[
        0
    ] = gyro_bias


    for index in range(
        number_of_epochs - 1
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


        if star_tracker_use_mask[
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

            nis_history[
                index + 1
            ] = (
                update[
                    "nis"
                ]
            )


        else:

            quaternion = (
                predicted_quaternion
            )

            gyro_bias = (
                predicted_bias
            )

            covariance = (
                predicted_covariance
            )


        quaternion_history[
            index + 1
        ] = quaternion

        rotation_history[
            index + 1
        ] = quaternion_to_rotation_matrix(
            quaternion
        )

        attitude_error_history[
            index + 1
        ] = (
            quaternion_attitude_error_angle(
                reference_quaternion=
                    true_quaternion_history[
                        index + 1
                    ],

                estimated_quaternion=
                    quaternion
            )
        )

        attitude_sigma_history[
            index + 1
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
            ])
        )

        bias_error_history[
            index + 1
        ] = np.linalg.norm(
            gyro_bias
            -
            true_gyro_bias_body
        )

        bias_estimate_history[
            index + 1
        ] = gyro_bias


    return {
        "quaternion_history":
            quaternion_history,

        "rotation_history":
            rotation_history,

        "attitude_error":
            attitude_error_history,

        "attitude_sigma":
            attitude_sigma_history,

        "bias_error":
            bias_error_history,

        "bias_estimate":
            bias_estimate_history,

        "nis":
            nis_history
    }


# ------------------------------------------------------------
# 12. EXECUTION ATTITUDE
# ------------------------------------------------------------

mekf_continuous = (
    run_mekf_case(
        star_tracker_use_mask=
            star_tracker_continuous_mask
    )
)

mekf_dual_outage = (
    run_mekf_case(
        star_tracker_use_mask=
            star_tracker_dual_outage_mask
    )
)


# ------------------------------------------------------------
# 13. ACCELEROMETRE
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
# 14. GNSS COMMUN
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
        number_of_epochs,
        3
    ),
    np.nan
)

gnss_covariances = np.full(
    (
        number_of_epochs,
        3,
        3
    ),
    np.nan
)

gnss_solution_available = np.zeros(
    number_of_epochs,
    dtype=bool
)

gnss_ls_errors = np.full(
    number_of_epochs,
    np.nan
)

pdop_values = np.full(
    number_of_epochs,
    np.nan
)

estimated_clock_bias_meters = (
    0.0
)


for index in range(
    1,
    number_of_epochs
):

    if not gnss_allowed[
        index
    ]:

        continue


    (
        all_gnss_positions,
        all_gnss_ids
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
            all_gnss_positions,

        satellite_ids=
            all_gnss_ids
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


    pseudoranges = (
        pseudorange_result[
            "pseudoranges"
        ]
    )


    gnss_initial_position = (
        truth_states[
            index,
            0:3
        ]
        +
        np.array([
            1000.0,
            -800.0,
            600.0
        ])
    )


    try:

        solution = (
            solve_gnss_least_squares_transmit_time(
                satellite_ids=
                    visible_ids,

                pseudoranges=
                    pseudoranges,

                reception_time_seconds=
                    time[
                        index
                    ],

                initial_position=
                    gnss_initial_position,

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
        ] = gnss_position

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
# 15. INITIALISATION NAVIGATION
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

initial_navigation_state = np.concatenate(
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

initial_navigation_covariance = np.diag([
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


# ------------------------------------------------------------
# 16. NAVIGATION GENERIQUE
# ------------------------------------------------------------

def run_navigation_case(
    rotation_history
):

    state = (
        initial_navigation_state.copy()
    )

    covariance = (
        initial_navigation_covariance.copy()
    )


    position_errors = np.zeros(
        number_of_epochs
    )

    position_sigmas = np.zeros(
        number_of_epochs
    )

    bias_errors = np.zeros(
        number_of_epochs
    )

    nis_history = np.full(
        number_of_epochs,
        np.nan
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

    bias_errors[
        0
    ] = np.linalg.norm(
        state[
            6:9
        ]
        -
        true_accelerometer_bias_body
    )


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
                    rotation_history[
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

            innovation_covariance = (
                update_result[
                    "innovation_covariance"
                ]
            )

            nis_history[
                index
            ] = (
                innovation.T
                @ np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )

            state = (
                update_result[
                    "state"
                ]
            )

            covariance = (
                update_result[
                    "covariance"
                ]
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

        bias_errors[
            index
        ] = np.linalg.norm(
            state[
                6:9
            ]
            -
            true_accelerometer_bias_body
        )


    return {
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "bias_errors":
            bias_errors,

        "nis":
            nis_history
    }


# ------------------------------------------------------------
# 17. EXECUTION NAVIGATION
# ------------------------------------------------------------

perfect_navigation = (
    run_navigation_case(
        rotation_history=
            true_rotation_history
    )
)

continuous_navigation = (
    run_navigation_case(
        rotation_history=
            mekf_continuous[
                "rotation_history"
            ]
    )
)

dual_outage_navigation = (
    run_navigation_case(
        rotation_history=
            mekf_dual_outage[
                "rotation_history"
            ]
    )
)


# ------------------------------------------------------------
# 18. INDICES IMPORTANTS
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]

outage_start_index = (
    outage_indices[
        0
    ]
)

outage_end_index = (
    outage_indices[
        -1
    ]
)

pre_outage_index = (
    outage_start_index
    - 1
)


# Premier update ST apres la coupure
post_outage_star_tracker_indices = np.where(
    star_tracker_dual_outage_mask
    &
    (
        time_minutes
        >
        outage_end_minutes
    )
)[0]

first_post_outage_star_tracker_index = (
    post_outage_star_tracker_indices[
        0
    ]
)


# ------------------------------------------------------------
# 19. STATISTIQUES ATTITUDE
# ------------------------------------------------------------

def compute_attitude_statistics(
    result,
    use_mask
):

    attitude_error = (
        result[
            "attitude_error"
        ]
    )

    attitude_sigma = (
        result[
            "attitude_sigma"
        ]
    )

    bias_error = (
        result[
            "bias_error"
        ]
    )

    nis = (
        result[
            "nis"
        ]
    )


    valid_nis = np.isfinite(
        nis
    )


    return {
        "rms_global":
            np.sqrt(
                np.mean(
                    attitude_error**2
                )
            ),

        "rms_outage":
            np.sqrt(
                np.mean(
                    attitude_error[
                        outage_mask
                    ]**2
                )
            ),

        "start_outage_error":
            attitude_error[
                outage_start_index
            ],

        "end_outage_error":
            attitude_error[
                outage_end_index
            ],

        "end_outage_sigma":
            attitude_sigma[
                outage_end_index
            ],

        "final_bias_error":
            bias_error[
                -1
            ],

        "mean_nis":
            np.mean(
                nis[
                    valid_nis
                ]
            ),

        "number_updates":
            np.sum(
                use_mask
            )
    }


continuous_attitude_stats = (
    compute_attitude_statistics(
        mekf_continuous,
        star_tracker_continuous_mask
    )
)

dual_attitude_stats = (
    compute_attitude_statistics(
        mekf_dual_outage,
        star_tracker_dual_outage_mask
    )
)


first_post_outage_star_tracker_nis = (
    mekf_dual_outage[
        "nis"
    ][
        first_post_outage_star_tracker_index
    ]
)


# ------------------------------------------------------------
# 20. STATISTIQUES NAVIGATION
# ------------------------------------------------------------

def compute_navigation_statistics(
    result
):

    position_errors = (
        result[
            "position_errors"
        ]
    )

    position_sigmas = (
        result[
            "position_sigmas"
        ]
    )

    bias_errors = (
        result[
            "bias_errors"
        ]
    )

    nis = (
        result[
            "nis"
        ]
    )

    valid_nis = np.isfinite(
        nis
    )


    return {
        "rmse_gnss":
            np.sqrt(
                np.mean(
                    position_errors[
                        gnss_solution_available
                    ]**2
                )
            ),

        "rmse_outage":
            np.sqrt(
                np.mean(
                    position_errors[
                        outage_mask
                    ]**2
                )
            ),

        "end_outage_error":
            position_errors[
                outage_end_index
            ],

        "end_outage_sigma":
            position_sigmas[
                outage_end_index
            ],

        "pre_outage_bias_error":
            bias_errors[
                pre_outage_index
            ],

        "mean_nis":
            np.mean(
                nis[
                    valid_nis
                ]
            )
    }


perfect_navigation_stats = (
    compute_navigation_statistics(
        perfect_navigation
    )
)

continuous_navigation_stats = (
    compute_navigation_statistics(
        continuous_navigation
    )
)

dual_navigation_stats = (
    compute_navigation_statistics(
        dual_outage_navigation
    )
)


dual_outage_navigation_degradation = (
    dual_navigation_stats[
        "rmse_outage"
    ]
    /
    continuous_navigation_stats[
        "rmse_outage"
    ]
)


# ------------------------------------------------------------
# 21. GNSS STATISTICS
# ------------------------------------------------------------

valid_gnss_ls = np.isfinite(
    gnss_ls_errors
)

valid_pdop = np.isfinite(
    pdop_values
)

gnss_ls_rmse = np.sqrt(
    np.mean(
        gnss_ls_errors[
            valid_gnss_ls
        ]**2
    )
)

mean_pdop = np.mean(
    pdop_values[
        valid_pdop
    ]
)


# ------------------------------------------------------------
# 22. PRINT
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================================================="
)

print(
    "AURORA — Experience 012-B"
)

print(
    "Perte simultanee GNSS + star tracker"
)

print(
    "===================================================================================================================================="
)

print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
)

print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)

print(
    f"Double outage : "
    f"{outage_start_minutes:.1f} "
    f"a {outage_end_minutes:.1f} min"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()


print(
    "----- GNSS -----"
)

print(
    f"RMSE GNSS LS : "
    f"{gnss_ls_rmse:.3f} m"
)

print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)

print()


print(
    "----- STAR TRACKER -----"
)

print(
    f"Periode : "
    f"{star_tracker_period_seconds:.1f} s"
)

print(
    f"Bruit sigma par axe : "
    f"{star_tracker_noise_std_deg:.4f} deg"
)

print(
    f"Corrections cas continu : "
    f"{continuous_attitude_stats['number_updates']}"
)

print(
    f"Corrections cas dual-outage : "
    f"{dual_attitude_stats['number_updates']}"
)

print()


print(
    "----- ATTITUDE -----"
)

header_attitude = (
    f"{'Architecture':>22} | "
    f"{'RMS global':>11} | "
    f"{'RMS outage':>11} | "
    f"{'Err debut':>10} | "
    f"{'Err fin':>10} | "
    f"{'Sigma fin':>10} | "
    f"{'Bias gyro fin':>13} | "
    f"{'NIS':>7}"
)

print(
    header_attitude
)

print(
    "-" * len(
        header_attitude
    )
)

print(
    f"{'MEKF ST continu':>22} | "
    f"{np.rad2deg(continuous_attitude_stats['rms_global']):11.4f} | "
    f"{np.rad2deg(continuous_attitude_stats['rms_outage']):11.4f} | "
    f"{np.rad2deg(continuous_attitude_stats['start_outage_error']):10.4f} | "
    f"{np.rad2deg(continuous_attitude_stats['end_outage_error']):10.4f} | "
    f"{np.rad2deg(continuous_attitude_stats['end_outage_sigma']):10.4f} | "
    f"{np.rad2deg(continuous_attitude_stats['final_bias_error']):13.6f} | "
    f"{continuous_attitude_stats['mean_nis']:7.3f}"
)

print(
    f"{'MEKF double outage':>22} | "
    f"{np.rad2deg(dual_attitude_stats['rms_global']):11.4f} | "
    f"{np.rad2deg(dual_attitude_stats['rms_outage']):11.4f} | "
    f"{np.rad2deg(dual_attitude_stats['start_outage_error']):10.4f} | "
    f"{np.rad2deg(dual_attitude_stats['end_outage_error']):10.4f} | "
    f"{np.rad2deg(dual_attitude_stats['end_outage_sigma']):10.4f} | "
    f"{np.rad2deg(dual_attitude_stats['final_bias_error']):13.6f} | "
    f"{dual_attitude_stats['mean_nis']:7.3f}"
)

print()

print(
    f"Premier NIS star tracker apres blackout : "
    f"{first_post_outage_star_tracker_nis:.3f}"
)

print(
    "Valeur theorique moyenne NIS attitude ~3"
)

print()


print(
    "----- NAVIGATION -----"
)

header_navigation = (
    f"{'Architecture':>22} | "
    f"{'Bias pre-outage':>17} | "
    f"{'RMSE GNSS':>10} | "
    f"{'RMSE outage':>12} | "
    f"{'Erreur fin':>11} | "
    f"{'Sigma fin':>10} | "
    f"{'NIS':>7}"
)

print(
    header_navigation
)

print(
    "-" * len(
        header_navigation
    )
)

print(
    f"{'Attitude parfaite':>22} | "
    f"{perfect_navigation_stats['pre_outage_bias_error']:17.6e} | "
    f"{perfect_navigation_stats['rmse_gnss']:10.3f} | "
    f"{perfect_navigation_stats['rmse_outage']:12.3f} | "
    f"{perfect_navigation_stats['end_outage_error']:11.3f} | "
    f"{perfect_navigation_stats['end_outage_sigma']:10.3f} | "
    f"{perfect_navigation_stats['mean_nis']:7.3f}"
)

print(
    f"{'MEKF ST continu':>22} | "
    f"{continuous_navigation_stats['pre_outage_bias_error']:17.6e} | "
    f"{continuous_navigation_stats['rmse_gnss']:10.3f} | "
    f"{continuous_navigation_stats['rmse_outage']:12.3f} | "
    f"{continuous_navigation_stats['end_outage_error']:11.3f} | "
    f"{continuous_navigation_stats['end_outage_sigma']:10.3f} | "
    f"{continuous_navigation_stats['mean_nis']:7.3f}"
)

print(
    f"{'MEKF double outage':>22} | "
    f"{dual_navigation_stats['pre_outage_bias_error']:17.6e} | "
    f"{dual_navigation_stats['rmse_gnss']:10.3f} | "
    f"{dual_navigation_stats['rmse_outage']:12.3f} | "
    f"{dual_navigation_stats['end_outage_error']:11.3f} | "
    f"{dual_navigation_stats['end_outage_sigma']:10.3f} | "
    f"{dual_navigation_stats['mean_nis']:7.3f}"
)

print()

print(
    f"Facteur degradation navigation dual / ST continu : "
    f"{dual_outage_navigation_degradation:.3f}"
)

print(
    "===================================================================================================================================="
)


# ------------------------------------------------------------
# 23. FIGURE ATTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_continuous[
            "attitude_error"
        ]
    ),
    label="MEKF — star tracker continu"
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_dual_outage[
            "attitude_error"
        ]
    ),
    label="MEKF — star tracker indisponible"
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_dual_outage[
            "attitude_sigma"
        ]
    ),
    linestyle="--",
    label="Sigma attitude 3D — dual outage"
)

plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15,
    label="GNSS + star tracker outage"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur attitude [deg]"
)

plt.title(
    "AURORA — Resilience attitude pendant perte star tracker"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. FIGURE NAVIGATION
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    time_minutes,
    perfect_navigation[
        "position_errors"
    ],
    label="Attitude parfaite"
)

plt.plot(
    time_minutes,
    continuous_navigation[
        "position_errors"
    ],
    label="MEKF — ST continu"
)

plt.plot(
    time_minutes,
    dual_outage_navigation[
        "position_errors"
    ],
    label="MEKF — double outage"
)

plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15,
    label="GNSS + star tracker outage"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur position 3D [m]"
)

plt.title(
    "AURORA — Navigation pendant perte simultanee GNSS + star tracker"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 25. ERREUR VS COVARIANCE
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    time_minutes,
    dual_outage_navigation[
        "position_errors"
    ],
    label="Erreur position — double outage"
)

plt.plot(
    time_minutes,
    dual_outage_navigation[
        "position_sigmas"
    ],
    linestyle="--",
    label="Sigma position 3D"
)

plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Position [m]"
)

plt.title(
    "AURORA — Coherence navigation pendant double outage"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. BIAIS GYRO
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)

plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_dual_outage[
            "bias_error"
        ]
    ),
    label="Erreur biais gyro — double outage"
)

plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur biais gyro [deg/s]"
)

plt.title(
    "AURORA — Stabilite du biais gyro pendant perte star tracker"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()