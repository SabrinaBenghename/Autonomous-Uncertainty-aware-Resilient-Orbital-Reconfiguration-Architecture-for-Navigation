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
    propagate_quaternion_with_body_rate,
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
# Experience 012-A
#
# Couplage attitude + navigation translationnelle
#
#
# Comparaison de trois architectures :
#
# 1. PERFECT ATTITUDE
#
#       R_BI = R_BI,true
#
#
# 2. GYRO ONLY
#
#       gyro
#        ↓
#       quaternion
#        ↓
#       R_BI,gyro
#
#
# 3. MEKF
#
#       gyro + star tracker
#              ↓
#             MEKF
#              ↓
#          R_BI,MEKF
#
#
# Puis dans les trois cas :
#
#     accelerometre BODY
#            ↓
#     rotation BODY -> ECI
#            ↓
#     EKF navigation 9D
#
#
# Etat navigation :
#
#     x_nav =
#         [r_I, v_I, b_a,B]
#
#
# Coupure GNSS :
#
#     60 -> 80 min
#
# Le star tracker reste disponible.
#
# Objectif :
#
# verifier que l'attitude estimee par le MEKF
# peut remplacer l'attitude vraie dans la
# navigation translationnelle.
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
# 3. COUPURE GNSS
# ------------------------------------------------------------

outage_start_minutes = (
    60.0
)


outage_end_minutes = (
    80.0
)


gnss_available = ~(
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


outage_mask = (
    ~gnss_available
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
# 6. VITESSE ANGULAIRE VRAIE
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


    star_tracker_available[
        index
    ] = (
        True
    )


# ------------------------------------------------------------
# 9. PROPAGATION ATTITUDE GYRO-ONLY
# ------------------------------------------------------------

gyro_only_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


gyro_only_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


gyro_only_attitude_error = np.zeros(
    number_of_epochs
)


gyro_only_quaternion_history[
    0
] = (
    true_quaternion_history[
        0
    ]
)


gyro_only_rotation_history[
    0
] = (
    true_rotation_history[
        0
    ]
)


for index in range(
    number_of_epochs - 1
):

    gyro_only_quaternion_history[
        index + 1
    ] = (
        propagate_quaternion_with_body_rate(
            quaternion_body_to_eci=
                gyro_only_quaternion_history[
                    index
                ],

            angular_rate_body=
                gyro_measurements[
                    index
                ],

            dt=
                dt
        )
    )


    gyro_only_rotation_history[
        index + 1
    ] = (
        quaternion_to_rotation_matrix(
            gyro_only_quaternion_history[
                index + 1
            ]
        )
    )


    gyro_only_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                gyro_only_quaternion_history[
                    index + 1
                ]
        )
    )


# ------------------------------------------------------------
# 10. PROPAGATION ATTITUDE MEKF
# ------------------------------------------------------------

mekf_quaternion_history = np.zeros(
    (
        number_of_epochs,
        4
    )
)


mekf_rotation_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


mekf_attitude_error = np.zeros(
    number_of_epochs
)


mekf_attitude_sigma = np.zeros(
    number_of_epochs
)


mekf_bias_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


mekf_bias_error = np.zeros(
    number_of_epochs
)


mekf_nis = np.full(
    number_of_epochs,
    np.nan
)


mekf_quaternion = (
    true_quaternion_history[
        0
    ].copy()
)


mekf_gyro_bias = np.zeros(
    3
)


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


mekf_quaternion_history[
    0
] = (
    mekf_quaternion
)


mekf_rotation_history[
    0
] = (
    quaternion_to_rotation_matrix(
        mekf_quaternion
    )
)


mekf_attitude_sigma[
    0
] = np.sqrt(
    np.trace(
        mekf_covariance[
            0:3,
            0:3
        ]
    )
)


mekf_bias_history[
    0
] = (
    mekf_gyro_bias
)


mekf_bias_error[
    0
] = np.linalg.norm(
    mekf_gyro_bias
    -
    true_gyro_bias_body
)


for index in range(
    number_of_epochs - 1
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


    predicted_gyro_bias = (
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
                    predicted_gyro_bias,

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


        mekf_nis[
            index + 1
        ] = (
            update[
                "nis"
            ]
        )


    else:

        mekf_quaternion = (
            predicted_quaternion
        )


        mekf_gyro_bias = (
            predicted_gyro_bias
        )


        mekf_covariance = (
            predicted_covariance
        )


    mekf_quaternion_history[
        index + 1
    ] = (
        mekf_quaternion
    )


    mekf_rotation_history[
        index + 1
    ] = (
        quaternion_to_rotation_matrix(
            mekf_quaternion
        )
    )


    mekf_attitude_error[
        index + 1
    ] = (
        quaternion_attitude_error_angle(
            reference_quaternion=
                true_quaternion_history[
                    index + 1
                ],

            estimated_quaternion=
                mekf_quaternion
        )
    )


    mekf_attitude_sigma[
        index + 1
    ] = np.sqrt(
        np.trace(
            mekf_covariance[
                0:3,
                0:3
            ]
        )
    )


    mekf_bias_history[
        index + 1
    ] = (
        mekf_gyro_bias
    )


    mekf_bias_error[
        index + 1
    ] = np.linalg.norm(
        mekf_gyro_bias
        -
        true_gyro_bias_body
    )


# ------------------------------------------------------------
# 11. ACCELEROMETRE PHYSIQUE BODY
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
# 12. GNSS COMMUN AUX TROIS FILTRES
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

    if not gnss_available[
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
# 13. INITIALISATION NAVIGATION
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
# 14. FONCTION GENERIQUE DE NAVIGATION
# ------------------------------------------------------------

def run_navigation_case(
    rotation_history
):
    """
    Execute le meme EKF navigation avec une
    histoire d'attitude donnee.

    Cela garantit que :

    - le GNSS est identique
    - l'accelerometre est identique
    - l'etat initial est identique
    - P0 est identique

    Seule l'attitude change.
    """

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


    accelerometer_bias_errors = np.zeros(
        number_of_epochs
    )


    nis_history = np.full(
        number_of_epochs,
        np.nan
    )


    # --------------------------------------------------------
    # Epoch 0
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


    accelerometer_bias_errors[
        0
    ] = np.linalg.norm(
        state[
            6:9
        ]
        -
        true_accelerometer_bias_body
    )


    # --------------------------------------------------------
    # Temporal loop
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ====================================================
        # A. PREDICTION
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
                    rotation_history[
                        index - 1
                    ],

                accelerometer_noise_std=
                    accelerometer_noise_std,

                bias_random_walk_density=
                    accelerometer_bias_random_walk_density
            )
        )


        # ====================================================
        # B. GNSS UPDATE
        # ====================================================

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


        # ====================================================
        # C. DIAGNOSTICS
        # ====================================================

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
        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "accelerometer_bias_errors":
            accelerometer_bias_errors,

        "nis":
            nis_history
    }


# ------------------------------------------------------------
# 15. EXECUTION DES TROIS ARCHITECTURES
# ------------------------------------------------------------

perfect_navigation = (
    run_navigation_case(
        rotation_history=
            true_rotation_history
    )
)


gyro_only_navigation = (
    run_navigation_case(
        rotation_history=
            gyro_only_rotation_history
    )
)


mekf_navigation = (
    run_navigation_case(
        rotation_history=
            mekf_rotation_history
    )
)


# ------------------------------------------------------------
# 16. INDICES IMPORTANTS
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


# ------------------------------------------------------------
# 17. FONCTION STATISTIQUE NAVIGATION
# ------------------------------------------------------------

def compute_navigation_statistics(
    navigation_result
):

    position_errors = (
        navigation_result[
            "position_errors"
        ]
    )


    position_sigmas = (
        navigation_result[
            "position_sigmas"
        ]
    )


    bias_errors = (
        navigation_result[
            "accelerometer_bias_errors"
        ]
    )


    nis_history = (
        navigation_result[
            "nis"
        ]
    )


    gnss_mask = (
        gnss_solution_available
    )


    valid_nis = np.isfinite(
        nis_history
    )


    rmse_gnss = np.sqrt(
        np.mean(
            position_errors[
                gnss_mask
            ]**2
        )
    )


    rmse_outage = np.sqrt(
        np.mean(
            position_errors[
                outage_mask
            ]**2
        )
    )


    mean_nis = np.mean(
        nis_history[
            valid_nis
        ]
    )


    return {
        "rmse_gnss":
            rmse_gnss,

        "rmse_outage":
            rmse_outage,

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
            mean_nis
    }


perfect_stats = (
    compute_navigation_statistics(
        perfect_navigation
    )
)


gyro_only_stats = (
    compute_navigation_statistics(
        gyro_only_navigation
    )
)


mekf_stats = (
    compute_navigation_statistics(
        mekf_navigation
    )
)


# ------------------------------------------------------------
# 18. ATTITUDE STATISTICS
# ------------------------------------------------------------

gyro_only_rms_attitude_error = np.sqrt(
    np.mean(
        gyro_only_attitude_error**2
    )
)


mekf_rms_attitude_error = np.sqrt(
    np.mean(
        mekf_attitude_error**2
    )
)


gyro_only_outage_rms_attitude_error = np.sqrt(
    np.mean(
        gyro_only_attitude_error[
            outage_mask
        ]**2
    )
)


mekf_outage_rms_attitude_error = np.sqrt(
    np.mean(
        mekf_attitude_error[
            outage_mask
        ]**2
    )
)


gyro_only_attitude_error_start_outage = (
    gyro_only_attitude_error[
        outage_start_index
    ]
)


gyro_only_attitude_error_end_outage = (
    gyro_only_attitude_error[
        outage_end_index
    ]
)


mekf_attitude_error_start_outage = (
    mekf_attitude_error[
        outage_start_index
    ]
)


mekf_attitude_error_end_outage = (
    mekf_attitude_error[
        outage_end_index
    ]
)


valid_mekf_nis = np.isfinite(
    mekf_nis
)


mean_mekf_nis = np.mean(
    mekf_nis[
        valid_mekf_nis
    ]
)


final_gyro_bias_error = (
    mekf_bias_error[
        -1
    ]
)


# ------------------------------------------------------------
# 19. GNSS STATISTICS
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
# 20. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=============================================================================================================================="
)


print(
    "AURORA — Experience 012-A"
)


print(
    "Couplage attitude estimee + navigation GNSS/accelerometre"
)


print(
    "=============================================================================================================================="
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
    f"Coupure GNSS : "
    f"{outage_start_minutes:.1f} "
    f"a {outage_end_minutes:.1f} min"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- GNSS COMMUN -----"
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
    "----- ATTITUDE -----"
)


print(
    f"Gyro-only RMS global : "
    f"{np.rad2deg(gyro_only_rms_attitude_error):.3f} deg"
)


print(
    f"MEKF RMS global : "
    f"{np.rad2deg(mekf_rms_attitude_error):.4f} deg"
)


print()


print(
    f"Gyro-only RMS pendant outage : "
    f"{np.rad2deg(gyro_only_outage_rms_attitude_error):.3f} deg"
)


print(
    f"MEKF RMS pendant outage : "
    f"{np.rad2deg(mekf_outage_rms_attitude_error):.4f} deg"
)


print()


print(
    f"Gyro-only erreur debut outage : "
    f"{np.rad2deg(gyro_only_attitude_error_start_outage):.3f} deg"
)


print(
    f"Gyro-only erreur fin outage : "
    f"{np.rad2deg(gyro_only_attitude_error_end_outage):.3f} deg"
)


print()


print(
    f"MEKF erreur debut outage : "
    f"{np.rad2deg(mekf_attitude_error_start_outage):.4f} deg"
)


print(
    f"MEKF erreur fin outage : "
    f"{np.rad2deg(mekf_attitude_error_end_outage):.4f} deg"
)


print()


print(
    f"Erreur finale biais gyro MEKF : "
    f"{np.rad2deg(final_gyro_bias_error):.6f} deg/s"
)


print(
    f"NIS attitude MEKF moyen : "
    f"{mean_mekf_nis:.3f}"
)


print(
    "Valeur theorique attitude NIS ~3"
)


print()


print(
    "----- NAVIGATION TRANSLATIONNELLE -----"
)


header = (
    f"{'Architecture':>18} | "
    f"{'Bias pre-outage':>17} | "
    f"{'RMSE GNSS':>10} | "
    f"{'RMSE outage':>12} | "
    f"{'Erreur fin':>11} | "
    f"{'Sigma fin':>10} | "
    f"{'NIS':>7}"
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
    f"{'Attitude parfaite':>18} | "
    f"{perfect_stats['pre_outage_bias_error']:17.6e} | "
    f"{perfect_stats['rmse_gnss']:10.3f} | "
    f"{perfect_stats['rmse_outage']:12.3f} | "
    f"{perfect_stats['end_outage_error']:11.3f} | "
    f"{perfect_stats['end_outage_sigma']:10.3f} | "
    f"{perfect_stats['mean_nis']:7.3f}"
)


print(
    f"{'Gyro only':>18} | "
    f"{gyro_only_stats['pre_outage_bias_error']:17.6e} | "
    f"{gyro_only_stats['rmse_gnss']:10.3f} | "
    f"{gyro_only_stats['rmse_outage']:12.3f} | "
    f"{gyro_only_stats['end_outage_error']:11.3f} | "
    f"{gyro_only_stats['end_outage_sigma']:10.3f} | "
    f"{gyro_only_stats['mean_nis']:7.3f}"
)


print(
    f"{'MEKF':>18} | "
    f"{mekf_stats['pre_outage_bias_error']:17.6e} | "
    f"{mekf_stats['rmse_gnss']:10.3f} | "
    f"{mekf_stats['rmse_outage']:12.3f} | "
    f"{mekf_stats['end_outage_error']:11.3f} | "
    f"{mekf_stats['end_outage_sigma']:10.3f} | "
    f"{mekf_stats['mean_nis']:7.3f}"
)


print(
    "=============================================================================================================================="
)


# ------------------------------------------------------------
# 21. FIGURE ATTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    np.rad2deg(
        gyro_only_attitude_error
    ),
    label="Gyro only"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_error
    ),
    label="MEKF gyro + star tracker"
)


plt.plot(
    time_minutes,
    np.rad2deg(
        mekf_attitude_sigma
    ),
    linestyle="--",
    label="Sigma attitude MEKF 3D"
)


plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15,
    label="Coupure GNSS"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur attitude [deg]"
)


plt.title(
    "AURORA — Attitude utilisee par la navigation"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 22. FIGURE NAVIGATION
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
    gyro_only_navigation[
        "position_errors"
    ],
    label="Gyro only"
)


plt.plot(
    time_minutes,
    mekf_navigation[
        "position_errors"
    ],
    label="MEKF"
)


plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15,
    label="Coupure GNSS"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur position 3D [m]"
)


plt.title(
    "AURORA — Impact de l'estimation d'attitude sur la navigation"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 23. FIGURE MEKF VS PERFECT
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    perfect_navigation[
        "position_errors"
    ],
    label="Erreur position — attitude parfaite"
)


plt.plot(
    time_minutes,
    mekf_navigation[
        "position_errors"
    ],
    label="Erreur position — attitude MEKF"
)


plt.plot(
    time_minutes,
    mekf_navigation[
        "position_sigmas"
    ],
    linestyle="--",
    label="Sigma position 3D — MEKF"
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
    "AURORA — Navigation couplee utilisant l'attitude estimee"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 24. FIGURE BIAIS ACCELEROMETRE
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    perfect_navigation[
        "accelerometer_bias_errors"
    ],
    label="Attitude parfaite"
)


plt.plot(
    time_minutes,
    gyro_only_navigation[
        "accelerometer_bias_errors"
    ],
    label="Gyro only"
)


plt.plot(
    time_minutes,
    mekf_navigation[
        "accelerometer_bias_errors"
    ],
    label="MEKF"
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
    "Erreur biais accelerometre BODY [m/s²]"
)


plt.title(
    "AURORA — Couplage attitude / estimation biais accelerometre"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()