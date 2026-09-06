import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    propagate_orbit_with_j2_and_drag_like
)

from src.navigation.gnss_ekf_interface import (
    C_LIGHT,
    generate_simplified_gps_positions,
    select_visible_gnss_satellites,
    solve_gnss_least_squares
)

from src.navigation.gnss_signal_propagation import (
    simulate_pseudoranges_with_transmit_time,
    solve_gnss_least_squares_transmit_time,
    compute_transmit_time_solution_covariance
)


# ============================================================
# AURORA
# Experience 009-D
#
# Impact du temps de propagation GNSS
#
# Meme pseudorange physique :
#
# 1. Solveur naif :
#       position GPS au temps de reception
#
# 2. Solveur corrige :
#       position GPS au temps d'emission
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


time_minutes = (
    time
    / 60.0
)


# ------------------------------------------------------------
# 3. VERITE ORBITALE
# ------------------------------------------------------------

truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            2.0e-5
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
# 4. GNSS
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


rng = np.random.default_rng(
    2026
)


# ------------------------------------------------------------
# 5. STOCKAGE
# ------------------------------------------------------------

naive_position_errors = np.full(
    number_of_epochs,
    np.nan
)


corrected_position_errors = np.full(
    number_of_epochs,
    np.nan
)


naive_clock_errors = np.full(
    number_of_epochs,
    np.nan
)


corrected_clock_errors = np.full(
    number_of_epochs,
    np.nan
)


travel_time_means = np.full(
    number_of_epochs,
    np.nan
)


satellite_displacement_means = np.full(
    number_of_epochs,
    np.nan
)


corrected_pdop = np.full(
    number_of_epochs,
    np.nan
)


visible_counts = np.zeros(
    number_of_epochs,
    dtype=int
)


corrected_position_sigmas = np.full(
    number_of_epochs,
    np.nan
)


# ------------------------------------------------------------
# 6. BOUCLE
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # --------------------------------------------------------
    # Positions GPS au temps de reception
    # --------------------------------------------------------

    (
        all_positions_rx,
        all_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[
                index
            ]
    )


    (
        visible_positions_rx,
        visible_ids
    ) = select_visible_gnss_satellites(
        receiver_position=
            truth_states[
                index,
                0:3
            ],

        satellite_positions=
            all_positions_rx,

        satellite_ids=
            all_ids
    )


    visible_counts[
        index
    ] = len(
        visible_ids
    )


    if len(
        visible_ids
    ) < 4:

        continue


    # --------------------------------------------------------
    # Generation physique avec satellite au temps d'emission
    # --------------------------------------------------------

    measurement_result = (
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
                rng
        )
    )


    pseudoranges = (
        measurement_result[
            "pseudoranges"
        ]
    )


    travel_time_means[
        index
    ] = np.mean(
        measurement_result[
            "travel_times"
        ]
    )


    satellite_displacement_means[
        index
    ] = np.mean(
        measurement_result[
            "satellite_displacements"
        ]
    )


    # --------------------------------------------------------
    # Meme initialisation pour les deux solveurs
    #
    # Ici on fait une experience de validation du modele.
    # L'offset n'influence pas la solution finale si
    # Gauss-Newton converge correctement.
    # --------------------------------------------------------

    initial_position = (
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


    # ========================================================
    # A. SOLVEUR NAIF
    #
    # Il recoit une mesure generee avec t_tx
    # mais modele les satellites a t_rx.
    # ========================================================

    try:

        naive_solution = (
            solve_gnss_least_squares(
                satellite_positions=
                    visible_positions_rx,

                pseudoranges=
                    pseudoranges,

                initial_position=
                    initial_position,

                initial_clock_bias_meters=
                    0.0
            )
        )


        if naive_solution[
            "converged"
        ]:

            naive_position_errors[
                index
            ] = np.linalg.norm(
                naive_solution[
                    "position"
                ]
                - truth_states[
                    index,
                    0:3
                ]
            )


            naive_clock_errors[
                index
            ] = (
                naive_solution[
                    "clock_bias_meters"
                ]
                - receiver_clock_bias_meters
            )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        pass


    # ========================================================
    # B. SOLVEUR CORRIGE
    # ========================================================

    try:

        corrected_solution = (
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
                    initial_position,

                initial_clock_bias_meters=
                    0.0
            )
        )


        if corrected_solution[
            "converged"
        ]:

            corrected_position_errors[
                index
            ] = np.linalg.norm(
                corrected_solution[
                    "position"
                ]
                - truth_states[
                    index,
                    0:3
                ]
            )


            corrected_clock_errors[
                index
            ] = (
                corrected_solution[
                    "clock_bias_meters"
                ]
                - receiver_clock_bias_meters
            )


            covariance_result = (
                compute_transmit_time_solution_covariance(
                    receiver_position=
                        corrected_solution[
                            "position"
                        ],

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


            corrected_pdop[
                index
            ] = (
                covariance_result[
                    "pdop"
                ]
            )


            corrected_position_sigmas[
                index
            ] = np.sqrt(
                np.trace(
                    covariance_result[
                        "position_covariance"
                    ]
                )
            )


    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError
    ):

        pass


# ------------------------------------------------------------
# 7. MASQUES
# ------------------------------------------------------------

valid_naive = np.isfinite(
    naive_position_errors
)


valid_corrected = np.isfinite(
    corrected_position_errors
)


valid_travel_time = np.isfinite(
    travel_time_means
)


valid_displacement = np.isfinite(
    satellite_displacement_means
)


valid_pdop = np.isfinite(
    corrected_pdop
)


# ------------------------------------------------------------
# 8. STATISTIQUES
# ------------------------------------------------------------

naive_rmse = np.sqrt(
    np.mean(
        naive_position_errors[
            valid_naive
        ]**2
    )
)


corrected_rmse = np.sqrt(
    np.mean(
        corrected_position_errors[
            valid_corrected
        ]**2
    )
)


mean_absolute_naive_clock_error = np.mean(
    np.abs(
        naive_clock_errors[
            valid_naive
        ]
    )
)


mean_absolute_corrected_clock_error = np.mean(
    np.abs(
        corrected_clock_errors[
            valid_corrected
        ]
    )
)


mean_travel_time = np.mean(
    travel_time_means[
        valid_travel_time
    ]
)


minimum_travel_time = np.min(
    measurement_result[
        "travel_times"
    ]
)


maximum_travel_time = np.max(
    measurement_result[
        "travel_times"
    ]
)


mean_satellite_displacement = np.mean(
    satellite_displacement_means[
        valid_displacement
    ]
)


mean_pdop = np.mean(
    corrected_pdop[
        valid_pdop
    ]
)


mean_position_sigma = np.mean(
    corrected_position_sigmas[
        np.isfinite(
            corrected_position_sigmas
        )
    ]
)


# ------------------------------------------------------------
# 9. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)


print(
    "AURORA — Experience 009-D"
)


print(
    "Impact du temps de propagation du signal GNSS"
)


print(
    "============================================================================================"
)


print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "----- PROPAGATION DU SIGNAL -----"
)


print(
    f"Temps de vol moyen : "
    f"{mean_travel_time * 1e3:.3f} ms"
)


print(
    f"Temps de vol min derniere epoque : "
    f"{minimum_travel_time * 1e3:.3f} ms"
)


print(
    f"Temps de vol max derniere epoque : "
    f"{maximum_travel_time * 1e3:.3f} ms"
)


print(
    f"Deplacement GPS moyen pendant le vol : "
    f"{mean_satellite_displacement:.3f} m"
)


print()


print(
    "----- SOLVEUR UTILISANT t_rx -----"
)


print(
    f"RMSE position : "
    f"{naive_rmse:.3f} m"
)


print(
    f"Erreur horloge absolue moyenne : "
    f"{mean_absolute_naive_clock_error:.3f} m"
)


print()


print(
    "----- SOLVEUR UTILISANT t_tx -----"
)


print(
    f"RMSE position : "
    f"{corrected_rmse:.3f} m"
)


print(
    f"Erreur horloge absolue moyenne : "
    f"{mean_absolute_corrected_clock_error:.3f} m"
)


print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)


print(
    f"Sigma position 3D moyen predit : "
    f"{mean_position_sigma:.3f} m"
)


print()


print(
    f"Facteur RMSE naif / corrige : "
    f"{naive_rmse / corrected_rmse:.2f}"
)


print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 10. FIGURE ERREUR POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    naive_position_errors,
    label="Modele satellite a t_rx"
)


plt.plot(
    time_minutes,
    corrected_position_errors,
    label="Modele satellite a t_tx"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur position 3D [m]"
)


plt.title(
    "AURORA — Effet du temps de propagation GNSS"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 11. FIGURE TEMPS DE VOL
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    travel_time_means
    * 1e3
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Temps de vol moyen [ms]"
)


plt.title(
    "AURORA — Temps de propagation des signaux GNSS"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 12. FIGURE DEPLACEMENT SATELLITE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    satellite_displacement_means
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Deplacement GPS [m]"
)


plt.title(
    "AURORA — Mouvement GPS pendant le temps de propagation"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()