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
    build_nadir_pointing_body_to_eci,
    body_to_eci
)

from src.sensors.body_frame_accelerometer import (
    simulate_body_frame_accelerometer_measurement
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_predict_with_bias_estimation,
    ekf_update_position_augmented
)

from src.navigation.body_bias_ekf import (
    ekf_predict_with_body_bias_estimation
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
# Experience 010-B
#
# Comparaison de deux modeles de biais accelerometre :
#
# A. ancien EKF :
#
#       x = [r, v, b_ECI]
#
#    suppose b_ECI constant
#
#
# B. nouvel EKF :
#
#       x = [r, v, b_BODY]
#
#    suppose b_BODY constant
#
#
# Le capteur physique possede un biais constant dans BODY.
#
# Attitude parfaite connue dans cette experience.
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

true_initial_position, true_initial_velocity = (
    keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )
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
# 5. ATTITUDE PARFAITE
# ------------------------------------------------------------

rotation_body_to_eci_history = np.zeros(
    (
        number_of_epochs,
        3,
        3
    )
)


for index in range(
    number_of_epochs
):

    rotation_body_to_eci_history[
        index
    ] = build_nadir_pointing_body_to_eci(
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


# ------------------------------------------------------------
# 6. ACCELEROMETRE PHYSIQUE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


true_bias_body = np.array([
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


accelerometer_measurements_eci = np.zeros(
    (
        number_of_epochs,
        3
    )
)


true_bias_eci_history = np.zeros(
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
                time[index],

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


    measurement_result = (
        simulate_body_frame_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force_eci,

            rotation_body_to_eci=
                rotation_body_to_eci_history[
                    index
                ],

            bias_body=
                true_bias_body,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    accelerometer_measurements_body[
        index
    ] = (
        measurement_result[
            "measurement_body"
        ]
    )


    accelerometer_measurements_eci[
        index
    ] = (
        measurement_result[
            "measurement_eci"
        ]
    )


    true_bias_eci_history[
        index
    ] = (
        measurement_result[
            "bias_eci"
        ]
    )


# ------------------------------------------------------------
# 7. GNSS
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


estimated_receiver_clock_bias_meters = (
    0.0
)


# ------------------------------------------------------------
# 8. ETATS INITIAUX
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


# Ancien modele :
#
# biais suppose constant dans ECI.

eci_bias_state = np.concatenate(
    (
        initial_orbital_estimate,

        np.zeros(
            3
        )
    )
)


# Nouveau modele :
#
# biais constant dans BODY.

body_bias_state = np.concatenate(
    (
        initial_orbital_estimate,

        np.zeros(
            3
        )
    )
)


# ------------------------------------------------------------
# 9. COVARIANCE INITIALE
# ------------------------------------------------------------

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)

initial_bias_sigma = (
    5.0e-6
)


P0 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2,

    initial_bias_sigma**2,
    initial_bias_sigma**2,
    initial_bias_sigma**2
])


eci_bias_covariance = (
    P0.copy()
)


body_bias_covariance = (
    P0.copy()
)


bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 10. STOCKAGE
# ------------------------------------------------------------

eci_position_errors = np.zeros(
    number_of_epochs
)


body_position_errors = np.zeros(
    number_of_epochs
)


eci_position_sigmas = np.zeros(
    number_of_epochs
)


body_position_sigmas = np.zeros(
    number_of_epochs
)


eci_bias_errors = np.zeros(
    number_of_epochs
)


body_bias_errors = np.zeros(
    number_of_epochs
)


eci_bias_estimates = np.zeros(
    (
        number_of_epochs,
        3
    )
)


body_bias_estimates = np.zeros(
    (
        number_of_epochs,
        3
    )
)


gnss_ls_errors = np.full(
    number_of_epochs,
    np.nan
)


pdop_values = np.full(
    number_of_epochs,
    np.nan
)


eci_nis = np.full(
    number_of_epochs,
    np.nan
)


body_nis = np.full(
    number_of_epochs,
    np.nan
)


visible_satellite_counts = np.zeros(
    number_of_epochs,
    dtype=int
)


# ------------------------------------------------------------
# 11. EPOQUE INITIALE
# ------------------------------------------------------------

eci_position_errors[
    0
] = np.linalg.norm(
    eci_bias_state[
        0:3
    ]
    - truth_states[
        0,
        0:3
    ]
)


body_position_errors[
    0
] = np.linalg.norm(
    body_bias_state[
        0:3
    ]
    - truth_states[
        0,
        0:3
    ]
)


eci_position_sigmas[
    0
] = np.sqrt(
    np.trace(
        eci_bias_covariance[
            0:3,
            0:3
        ]
    )
)


body_position_sigmas[
    0
] = np.sqrt(
    np.trace(
        body_bias_covariance[
            0:3,
            0:3
        ]
    )
)


eci_bias_errors[
    0
] = np.linalg.norm(
    eci_bias_state[
        6:9
    ]
    - true_bias_eci_history[
        0
    ]
)


body_bias_errors[
    0
] = np.linalg.norm(
    body_bias_state[
        6:9
    ]
    - true_bias_body
)


eci_bias_estimates[
    0
] = (
    eci_bias_state[
        6:9
    ]
)


body_bias_estimates[
    0
] = (
    body_bias_state[
        6:9
    ]
)


# ------------------------------------------------------------
# 12. BOUCLE DE NAVIGATION
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # ========================================================
    # A. ANCIEN EKF : BIAIS CONSTANT ECI
    # ========================================================

    (
        predicted_eci_state,
        predicted_eci_covariance
    ) = ekf_predict_with_bias_estimation(
        state=
            eci_bias_state,

        covariance=
            eci_bias_covariance,

        dt=
            dt,

        accelerometer_measurement_eci=
            accelerometer_measurements_eci[
                index - 1
            ],

        accelerometer_noise_std=
            accelerometer_noise_std,

        bias_random_walk_density=
            bias_random_walk_density
    )


    # ========================================================
    # B. NOUVEL EKF : BIAIS CONSTANT BODY
    # ========================================================

    (
        predicted_body_state,
        predicted_body_covariance
    ) = ekf_predict_with_body_bias_estimation(
        state=
            body_bias_state,

        covariance=
            body_bias_covariance,

        dt=
            dt,

        accelerometer_measurement_body=
            accelerometer_measurements_body[
                index - 1
            ],

        rotation_body_to_eci=
            rotation_body_to_eci_history[
                index - 1
            ],

        accelerometer_noise_std=
            accelerometer_noise_std,

        bias_random_walk_density=
            bias_random_walk_density
    )


    # ========================================================
    # C. GNSS
    # ========================================================

    if gnss_available[
        index
    ]:

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


        visible_satellite_counts[
            index
        ] = len(
            visible_ids
        )


        if len(
            visible_ids
        ) >= 4:

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


            # ------------------------------------------------
            # Une solution GNSS commune aux deux EKF
            #
            # Initialisation symetrique :
            # moyenne des deux predictions.
            # ------------------------------------------------

            common_initial_position = (
                0.5
                * (
                    predicted_eci_state[
                        0:3
                    ]
                    +
                    predicted_body_state[
                        0:3
                    ]
                )
            )


            try:

                gnss_solution = (
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
                            common_initial_position,

                        initial_clock_bias_meters=
                            estimated_receiver_clock_bias_meters
                    )
                )


                if gnss_solution[
                    "converged"
                ]:

                    gnss_position = (
                        gnss_solution[
                            "position"
                        ]
                    )


                    estimated_receiver_clock_bias_meters = (
                        gnss_solution[
                            "clock_bias_meters"
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


                    R_gnss = (
                        covariance_result[
                            "position_covariance"
                        ]
                    )


                    pdop_values[
                        index
                    ] = (
                        covariance_result[
                            "pdop"
                        ]
                    )


                    gnss_ls_errors[
                        index
                    ] = np.linalg.norm(
                        gnss_position
                        - truth_states[
                            index,
                            0:3
                        ]
                    )


                    # ========================================
                    # Update ancien EKF
                    # ========================================

                    eci_update = (
                        ekf_update_position_augmented(
                            predicted_state=
                                predicted_eci_state,

                            predicted_covariance=
                                predicted_eci_covariance,

                            position_measurement=
                                gnss_position,

                            measurement_noise_covariance=
                                R_gnss
                        )
                    )


                    eci_innovation = (
                        eci_update[
                            "innovation"
                        ]
                    )


                    eci_S = (
                        eci_update[
                            "innovation_covariance"
                        ]
                    )


                    eci_nis[
                        index
                    ] = (
                        eci_innovation.T
                        @ np.linalg.solve(
                            eci_S,
                            eci_innovation
                        )
                    )


                    eci_bias_state = (
                        eci_update[
                            "state"
                        ]
                    )


                    eci_bias_covariance = (
                        eci_update[
                            "covariance"
                        ]
                    )


                    # ========================================
                    # Update nouvel EKF BODY
                    # ========================================

                    body_update = (
                        ekf_update_position_augmented(
                            predicted_state=
                                predicted_body_state,

                            predicted_covariance=
                                predicted_body_covariance,

                            position_measurement=
                                gnss_position,

                            measurement_noise_covariance=
                                R_gnss
                        )
                    )


                    body_innovation = (
                        body_update[
                            "innovation"
                        ]
                    )


                    body_S = (
                        body_update[
                            "innovation_covariance"
                        ]
                    )


                    body_nis[
                        index
                    ] = (
                        body_innovation.T
                        @ np.linalg.solve(
                            body_S,
                            body_innovation
                        )
                    )


                    body_bias_state = (
                        body_update[
                            "state"
                        ]
                    )


                    body_bias_covariance = (
                        body_update[
                            "covariance"
                        ]
                    )


                else:

                    eci_bias_state = (
                        predicted_eci_state
                    )

                    eci_bias_covariance = (
                        predicted_eci_covariance
                    )


                    body_bias_state = (
                        predicted_body_state
                    )

                    body_bias_covariance = (
                        predicted_body_covariance
                    )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError
            ):

                eci_bias_state = (
                    predicted_eci_state
                )

                eci_bias_covariance = (
                    predicted_eci_covariance
                )


                body_bias_state = (
                    predicted_body_state
                )

                body_bias_covariance = (
                    predicted_body_covariance
                )


        else:

            eci_bias_state = (
                predicted_eci_state
            )

            eci_bias_covariance = (
                predicted_eci_covariance
            )


            body_bias_state = (
                predicted_body_state
            )

            body_bias_covariance = (
                predicted_body_covariance
            )


    # ========================================================
    # D. COUPURE GNSS
    # ========================================================

    else:

        eci_bias_state = (
            predicted_eci_state
        )

        eci_bias_covariance = (
            predicted_eci_covariance
        )


        body_bias_state = (
            predicted_body_state
        )

        body_bias_covariance = (
            predicted_body_covariance
        )


    # ========================================================
    # E. ERREURS
    # ========================================================

    eci_position_errors[
        index
    ] = np.linalg.norm(
        eci_bias_state[
            0:3
        ]
        - truth_states[
            index,
            0:3
        ]
    )


    body_position_errors[
        index
    ] = np.linalg.norm(
        body_bias_state[
            0:3
        ]
        - truth_states[
            index,
            0:3
        ]
    )


    eci_position_sigmas[
        index
    ] = np.sqrt(
        np.trace(
            eci_bias_covariance[
                0:3,
                0:3
            ]
        )
    )


    body_position_sigmas[
        index
    ] = np.sqrt(
        np.trace(
            body_bias_covariance[
                0:3,
                0:3
            ]
        )
    )


    # --------------------------------------------------------
    # Ancien EKF :
    # comparer son biais ECI estime au vrai biais ECI(t)
    # --------------------------------------------------------

    eci_bias_errors[
        index
    ] = np.linalg.norm(
        eci_bias_state[
            6:9
        ]
        - true_bias_eci_history[
            index
        ]
    )


    # --------------------------------------------------------
    # Nouvel EKF :
    # comparer directement dans BODY.
    # --------------------------------------------------------

    body_bias_errors[
        index
    ] = np.linalg.norm(
        body_bias_state[
            6:9
        ]
        - true_bias_body
    )


    eci_bias_estimates[
        index
    ] = (
        eci_bias_state[
            6:9
        ]
    )


    body_bias_estimates[
        index
    ] = (
        body_bias_state[
            6:9
        ]
    )


# ------------------------------------------------------------
# 13. INDICES
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
# 14. MASQUES
# ------------------------------------------------------------

evaluation_mask = np.ones(
    number_of_epochs,
    dtype=bool
)


evaluation_mask[
    0
] = False


gnss_evaluation_mask = (
    gnss_available
    &
    evaluation_mask
)


valid_gnss_ls = np.isfinite(
    gnss_ls_errors
)


valid_pdop = np.isfinite(
    pdop_values
)


valid_eci_nis = np.isfinite(
    eci_nis
)


valid_body_nis = np.isfinite(
    body_nis
)


# ------------------------------------------------------------
# 15. STATISTIQUES
# ------------------------------------------------------------

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


eci_rmse_gnss = np.sqrt(
    np.mean(
        eci_position_errors[
            gnss_evaluation_mask
        ]**2
    )
)


body_rmse_gnss = np.sqrt(
    np.mean(
        body_position_errors[
            gnss_evaluation_mask
        ]**2
    )
)


eci_rmse_outage = np.sqrt(
    np.mean(
        eci_position_errors[
            outage_mask
        ]**2
    )
)


body_rmse_outage = np.sqrt(
    np.mean(
        body_position_errors[
            outage_mask
        ]**2
    )
)


eci_end_outage_error = (
    eci_position_errors[
        outage_end_index
    ]
)


body_end_outage_error = (
    body_position_errors[
        outage_end_index
    ]
)


eci_end_outage_sigma = (
    eci_position_sigmas[
        outage_end_index
    ]
)


body_end_outage_sigma = (
    body_position_sigmas[
        outage_end_index
    ]
)


eci_bias_error_before_outage = (
    eci_bias_errors[
        pre_outage_index
    ]
)


body_bias_error_before_outage = (
    body_bias_errors[
        pre_outage_index
    ]
)


mean_eci_nis = np.mean(
    eci_nis[
        valid_eci_nis
    ]
)


mean_body_nis = np.mean(
    body_nis[
        valid_body_nis
    ]
)


# ------------------------------------------------------------
# 16. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================="
)


print(
    "AURORA — Experience 010-B"
)


print(
    "Comparaison biais constant ECI vs biais constant BODY"
)


print(
    "===================================================================================================="
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
    "----- GNSS -----"
)


print(
    f"RMSE position GNSS LS : "
    f"{gnss_ls_rmse:.3f} m"
)


print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)


print()


print(
    "----- BIAIS AVANT COUPURE -----"
)


print(
    f"Erreur biais ancien modele ECI : "
    f"{eci_bias_error_before_outage:.6e} m/s^2"
)


print(
    f"Erreur biais nouveau modele BODY : "
    f"{body_bias_error_before_outage:.6e} m/s^2"
)


print()


print(
    "----- GNSS DISPONIBLE -----"
)


print(
    f"RMSE position ancien modele ECI : "
    f"{eci_rmse_gnss:.3f} m"
)


print(
    f"RMSE position nouveau modele BODY : "
    f"{body_rmse_gnss:.3f} m"
)


print()


print(
    "----- COUPURE GNSS -----"
)


print(
    f"RMSE ancien modele ECI : "
    f"{eci_rmse_outage:.3f} m"
)


print(
    f"RMSE nouveau modele BODY : "
    f"{body_rmse_outage:.3f} m"
)


print()


print(
    f"Erreur fin coupure ancien modele ECI : "
    f"{eci_end_outage_error:.3f} m"
)


print(
    f"Sigma 3D fin coupure ancien modele ECI : "
    f"{eci_end_outage_sigma:.3f} m"
)


print()


print(
    f"Erreur fin coupure nouveau modele BODY : "
    f"{body_end_outage_error:.3f} m"
)


print(
    f"Sigma 3D fin coupure nouveau modele BODY : "
    f"{body_end_outage_sigma:.3f} m"
)


print()


print(
    "----- NIS -----"
)


print(
    f"NIS moyen ancien modele ECI : "
    f"{mean_eci_nis:.3f}"
)


print(
    f"NIS moyen nouveau modele BODY : "
    f"{mean_body_nis:.3f}"
)


print(
    "Valeur theorique attendue ~3"
)


print(
    "===================================================================================================="
)


# ------------------------------------------------------------
# 17. FIGURE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    eci_position_errors,
    label="Biais suppose constant ECI"
)


plt.plot(
    time_minutes,
    body_position_errors,
    label="Biais estime constant BODY"
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
    "AURORA — Impact du repere du biais accelerometre"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 18. FIGURE BIAIS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    eci_bias_errors,
    label="Erreur biais modele ECI"
)


plt.plot(
    time_minutes,
    body_bias_errors,
    label="Erreur biais modele BODY"
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
    "Norme erreur biais [m/s²]"
)


plt.title(
    "AURORA — Estimation du biais accelerometre"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 19. FIGURE ERREUR / COVARIANCE BODY
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    body_position_errors,
    label="Erreur position BODY-EKF"
)


plt.plot(
    time_minutes,
    body_position_sigmas,
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
    "AURORA — Coherence du nouvel EKF avec biais BODY"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()