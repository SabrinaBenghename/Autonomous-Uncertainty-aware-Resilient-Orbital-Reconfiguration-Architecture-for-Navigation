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

from src.sensors.accelerometer import (
    simulate_accelerometer_measurement
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_predict_with_bias_estimation,
    ekf_update_position_augmented
)

from src.navigation.gnss_ekf_interface import (
    C_LIGHT,
    generate_simplified_gps_positions,
    select_visible_gnss_satellites,
    simulate_pseudorange_measurements,
    solve_gnss_least_squares,
    compute_gnss_solution_covariance
)


# ============================================================
# AURORA
# Experience 009-A
#
# Connexion de la vraie chaine GNSS simulee
# au filtre EKF 9D.
#
# Chaine :
#
# constellation
#   -> visibilite
#   -> pseudodistances
#   -> estimation LS [x,y,z,b]
#   -> covariance GNSS
#   -> EKF 9D
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE CUBESAT
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

gnss_outage_start_minutes = (
    60.0
)

gnss_outage_end_minutes = (
    80.0
)


forced_gnss_available = ~(
    (
        time_minutes
        >= gnss_outage_start_minutes
    )
    &
    (
        time_minutes
        <= gnss_outage_end_minutes
    )
)


outage_mask = (
    ~forced_gnss_available
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


true_initial_state = np.concatenate(
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
            true_initial_state,

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
# 5. ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


true_accelerometer_bias = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])


accelerometer_rng = (
    np.random.default_rng(
        4242
    )
)


accelerometer_measurements = np.zeros(
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
        simulate_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force,

            bias_eci=
                true_accelerometer_bias,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    accelerometer_measurements[
        index
    ] = (
        measurement_result[
            "measurement"
        ]
    )


# ------------------------------------------------------------
# 6. GNSS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


receiver_clock_bias_meters = (
    C_LIGHT
    * receiver_clock_bias_seconds
)


gnss_rng = (
    np.random.default_rng(
        2026
    )
)


# ------------------------------------------------------------
# 7. ETAT INITIAL EKF 9D
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


initial_bias_estimate = np.zeros(
    3
)


estimated_state = np.concatenate(
    (
        true_initial_state
        +
        np.concatenate(
            (
                initial_position_error,
                initial_velocity_error
            )
        ),

        initial_bias_estimate
    )
)


# ------------------------------------------------------------
# 8. COVARIANCE INITIALE
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


covariance = np.diag([
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


bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 9. HORLOGE GNSS ESTIMEE
#
# Le biais horloge n'est pas encore dans l'EKF 9D.
# Il est estime par le solveur GNSS LS.
# ------------------------------------------------------------

estimated_receiver_clock_bias_meters = (
    0.0
)


# ------------------------------------------------------------
# 10. STOCKAGE
# ------------------------------------------------------------

estimated_states = np.zeros(
    (
        number_of_epochs,
        9
    )
)


position_errors_ekf = np.full(
    number_of_epochs,
    np.nan
)


position_sigma_ekf = np.full(
    number_of_epochs,
    np.nan
)


gnss_ls_position_errors = np.full(
    number_of_epochs,
    np.nan
)


gnss_position_sigma = np.full(
    number_of_epochs,
    np.nan
)


visible_satellite_counts = np.zeros(
    number_of_epochs,
    dtype=int
)


pdop_values = np.full(
    number_of_epochs,
    np.nan
)


nis_values = np.full(
    number_of_epochs,
    np.nan
)


bias_error_norms = np.full(
    number_of_epochs,
    np.nan
)


gnss_solution_available = np.zeros(
    number_of_epochs,
    dtype=bool
)


gnss_converged = np.zeros(
    number_of_epochs,
    dtype=bool
)


gnss_iterations = np.full(
    number_of_epochs,
    np.nan
)


# ------------------------------------------------------------
# 11. EPOQUE INITIALE
# ------------------------------------------------------------

estimated_states[
    0
] = (
    estimated_state
)


initial_augmented_truth = np.concatenate(
    (
        truth_states[
            0
        ],

        true_accelerometer_bias
    )
)


initial_estimation_error = (
    estimated_state
    - initial_augmented_truth
)


position_errors_ekf[
    0
] = np.linalg.norm(
    initial_estimation_error[
        0:3
    ]
)


position_sigma_ekf[
    0
] = np.sqrt(
    np.trace(
        covariance[
            0:3,
            0:3
        ]
    )
)


bias_error_norms[
    0
] = np.linalg.norm(
    estimated_state[
        6:9
    ]
    - true_accelerometer_bias
)


# ------------------------------------------------------------
# 12. BOUCLE DE NAVIGATION
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # ========================================================
    # A. PREDICTION EKF AVEC ACCELEROMETRE
    # ========================================================

    (
        predicted_state,
        predicted_covariance
    ) = ekf_predict_with_bias_estimation(
        state=
            estimated_state,

        covariance=
            covariance,

        dt=
            dt,

        accelerometer_measurement_eci=
            accelerometer_measurements[
                index - 1
            ],

        accelerometer_noise_std=
            accelerometer_noise_std,

        bias_random_walk_density=
            bias_random_walk_density
    )


    # ========================================================
    # B. CONSTELLATION GNSS
    # ========================================================

    (
        all_gnss_positions,
        all_gnss_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[
                index
            ]
    )


    # ========================================================
    # C. VISIBILITE PHYSIQUE
    #
    # La generation des signaux utilise la vraie
    # position du recepteur.
    # ========================================================

    (
        visible_positions,
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
        visible_positions
    )


    # ========================================================
    # D. GNSS DISPONIBLE ?
    # ========================================================

    can_use_gnss = (
        forced_gnss_available[
            index
        ]
        and
        len(
            visible_positions
        ) >= 4
    )


    # ========================================================
    # E. GENERATION PSEUDODISTANCES
    # ========================================================

    if can_use_gnss:

        pseudorange_result = (
            simulate_pseudorange_measurements(
                receiver_position=
                    truth_states[
                        index,
                        0:3
                    ],

                satellite_positions=
                    visible_positions,

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


        # ====================================================
        # F. SOLUTION GNSS PAR MOINDRES CARRES
        #
        # IMPORTANT :
        # l'initialisation vient de la prediction EKF,
        # pas de la position vraie.
        # ====================================================

        try:

            gnss_result = (
                solve_gnss_least_squares(
                    satellite_positions=
                        visible_positions,

                    pseudoranges=
                        pseudoranges,

                    initial_position=
                        predicted_state[
                            0:3
                        ],

                    initial_clock_bias_meters=
                        estimated_receiver_clock_bias_meters
                )
            )


            gnss_converged[
                index
            ] = (
                gnss_result[
                    "converged"
                ]
            )


            gnss_iterations[
                index
            ] = (
                gnss_result[
                    "iterations"
                ]
            )


            if gnss_result[
                "converged"
            ]:

                gnss_position = (
                    gnss_result[
                        "position"
                    ]
                )


                estimated_receiver_clock_bias_meters = (
                    gnss_result[
                        "clock_bias_meters"
                    ]
                )


                # ============================================
                # G. COVARIANCE GNSS
                # ============================================

                covariance_result = (
                    compute_gnss_solution_covariance(
                        receiver_position=
                            gnss_position,

                        satellite_positions=
                            visible_positions,

                        pseudorange_noise_std=
                            pseudorange_noise_std
                    )
                )


                R_gnss_position = (
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


                gnss_position_sigma[
                    index
                ] = np.sqrt(
                    np.trace(
                        R_gnss_position
                    )
                )


                gnss_ls_position_errors[
                    index
                ] = np.linalg.norm(
                    gnss_position
                    - truth_states[
                        index,
                        0:3
                    ]
                )


                # ============================================
                # H. CORRECTION EKF 9D
                # ============================================

                update_result = (
                    ekf_update_position_augmented(
                        predicted_state=
                            predicted_state,

                        predicted_covariance=
                            predicted_covariance,

                        position_measurement=
                            gnss_position,

                        measurement_noise_covariance=
                            R_gnss_position
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


                nis_values[
                    index
                ] = (
                    innovation.T
                    @ np.linalg.solve(
                        innovation_covariance,
                        innovation
                    )
                )


                estimated_state = (
                    update_result[
                        "state"
                    ]
                )


                covariance = (
                    update_result[
                        "covariance"
                    ]
                )


                gnss_solution_available[
                    index
                ] = (
                    True
                )


            else:

                estimated_state = (
                    predicted_state
                )

                covariance = (
                    predicted_covariance
                )


        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError
        ):

            estimated_state = (
                predicted_state
            )

            covariance = (
                predicted_covariance
            )


    else:

        estimated_state = (
            predicted_state
        )

        covariance = (
            predicted_covariance
        )


    # ========================================================
    # I. ERREURS EKF
    # ========================================================

    true_augmented_state = np.concatenate(
        (
            truth_states[
                index
            ],

            true_accelerometer_bias
        )
    )


    estimation_error = (
        estimated_state
        - true_augmented_state
    )


    estimated_states[
        index
    ] = (
        estimated_state
    )


    position_errors_ekf[
        index
    ] = np.linalg.norm(
        estimation_error[
            0:3
        ]
    )


    position_sigma_ekf[
        index
    ] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    bias_error_norms[
        index
    ] = np.linalg.norm(
        estimated_state[
            6:9
        ]
        - true_accelerometer_bias
    )


# ------------------------------------------------------------
# 13. MASQUES
# ------------------------------------------------------------

evaluation_mask = np.ones(
    number_of_epochs,
    dtype=bool
)

evaluation_mask[
    0
] = False


available_solution_mask = (
    gnss_solution_available
    & evaluation_mask
)


valid_gnss_ls_mask = np.isfinite(
    gnss_ls_position_errors
)


valid_pdop_mask = np.isfinite(
    pdop_values
)


valid_nis_mask = np.isfinite(
    nis_values
)


# ------------------------------------------------------------
# 14. STATISTIQUES GNSS BRUT
# ------------------------------------------------------------

gnss_ls_rmse = np.sqrt(
    np.mean(
        gnss_ls_position_errors[
            valid_gnss_ls_mask
        ]**2
    )
)


mean_pdop = np.mean(
    pdop_values[
        valid_pdop_mask
    ]
)


minimum_pdop = np.min(
    pdop_values[
        valid_pdop_mask
    ]
)


maximum_pdop = np.max(
    pdop_values[
        valid_pdop_mask
    ]
)


mean_visible_satellites = np.mean(
    visible_satellite_counts[
        evaluation_mask
    ]
)


minimum_visible_satellites = np.min(
    visible_satellite_counts[
        evaluation_mask
    ]
)


maximum_visible_satellites = np.max(
    visible_satellite_counts[
        evaluation_mask
    ]
)


gnss_convergence_rate = (
    100.0
    * np.sum(
        gnss_solution_available
    )
    / np.sum(
        forced_gnss_available
        & evaluation_mask
    )
)


# ------------------------------------------------------------
# 15. STATISTIQUES EKF
# ------------------------------------------------------------

ekf_rmse_gnss_available = np.sqrt(
    np.mean(
        position_errors_ekf[
            available_solution_mask
        ]**2
    )
)


ekf_rmse_outage = np.sqrt(
    np.mean(
        position_errors_ekf[
            outage_mask
        ]**2
    )
)


outage_indices = np.where(
    outage_mask
)[0]


outage_end_index = (
    outage_indices[
        -1
    ]
)


ekf_error_end_outage = (
    position_errors_ekf[
        outage_end_index
    ]
)


ekf_sigma_end_outage = (
    position_sigma_ekf[
        outage_end_index
    ]
)


mean_nis = np.mean(
    nis_values[
        valid_nis_mask
    ]
)


# ------------------------------------------------------------
# 16. BIAIS ACCELEROMETRE AVANT OUTAGE
# ------------------------------------------------------------

pre_outage_index = (
    outage_indices[
        0
    ]
    - 1
)


bias_error_before_outage = (
    bias_error_norms[
        pre_outage_index
    ]
)


# ------------------------------------------------------------
# 17. HORLOGE GNSS
# ------------------------------------------------------------

clock_bias_error_meters = (
    estimated_receiver_clock_bias_meters
    - receiver_clock_bias_meters
)


clock_bias_error_nanoseconds = (
    clock_bias_error_meters
    / C_LIGHT
    * 1e9
)


# ------------------------------------------------------------
# 18. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)

print(
    "AURORA — Experience 009-A"
)

print(
    "Chaine GNSS pseudorange -> LS -> covariance -> EKF 9D"
)

print(
    "============================================================================================"
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
    f"{gnss_outage_start_minutes:.1f} "
    f"a {gnss_outage_end_minutes:.1f} min"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()

print(
    "----- CONSTELLATION / GEOMETRIE -----"
)

print(
    f"Satellites visibles min : "
    f"{minimum_visible_satellites}"
)

print(
    f"Satellites visibles max : "
    f"{maximum_visible_satellites}"
)

print(
    f"Satellites visibles moyen : "
    f"{mean_visible_satellites:.2f}"
)

print(
    f"PDOP min : "
    f"{minimum_pdop:.3f}"
)

print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)

print(
    f"PDOP max : "
    f"{maximum_pdop:.3f}"
)

print()

print(
    "----- POSITIONNEMENT GNSS LS -----"
)

print(
    f"Taux de solutions GNSS disponibles : "
    f"{gnss_convergence_rate:.2f} %"
)

print(
    f"RMSE position GNSS LS : "
    f"{gnss_ls_rmse:.3f} m"
)

print(
    f"Biais horloge vrai : "
    f"{receiver_clock_bias_meters:.3f} m"
)

print(
    f"Biais horloge estime final : "
    f"{estimated_receiver_clock_bias_meters:.3f} m"
)

print(
    f"Erreur horloge finale : "
    f"{clock_bias_error_meters:.3f} m "
    f"({clock_bias_error_nanoseconds:.3f} ns)"
)

print()

print(
    "----- EKF 9D -----"
)

print(
    f"RMSE position avec GNSS disponible : "
    f"{ekf_rmse_gnss_available:.3f} m"
)

print(
    f"RMSE pendant coupure GNSS : "
    f"{ekf_rmse_outage:.3f} m"
)

print(
    f"Erreur fin coupure : "
    f"{ekf_error_end_outage:.3f} m"
)

print(
    f"Incertitude position 3D fin coupure : "
    f"{ekf_sigma_end_outage:.3f} m"
)

print(
    f"NIS moyen : "
    f"{mean_nis:.3f} "
    f"(attendu ~3)"
)

print()

print(
    "----- BIAIS ACCELEROMETRE -----"
)

print(
    f"Erreur biais juste avant coupure : "
    f"{bias_error_before_outage:.6e} m/s^2"
)

print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 19. FIGURE GNSS LS VS EKF
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    gnss_ls_position_errors,
    label="Erreur position GNSS LS"
)

plt.plot(
    time_minutes,
    position_errors_ekf,
    label="Erreur position EKF 9D"
)

plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
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
    "AURORA — GNSS LS vs navigation EKF"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. FIGURE ERREUR / INCERTITUDE EKF
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    position_errors_ekf,
    label="Erreur EKF"
)

plt.plot(
    time_minutes,
    position_sigma_ekf,
    linestyle="--",
    label="Incertitude position 3D"
)

plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Position [m]"
)

plt.title(
    "AURORA — Erreur et covariance de navigation"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. FIGURE PDOP
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    pdop_values
)

plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "PDOP [-]"
)

plt.title(
    "AURORA — Geometrie GNSS utilisee par l'EKF"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. FIGURE NOMBRE DE SATELLITES
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    visible_satellite_counts
)

plt.axhline(
    4,
    linestyle="--",
    label="Minimum pour positionnement"
)

plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Satellites visibles"
)

plt.title(
    "AURORA — Disponibilite geometrique GNSS"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()