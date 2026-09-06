import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH,
    propagate_orbit
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.navigation.gnss_constellation import (
    generate_simplified_gps_constellation,
    propagate_gnss_constellation
)

from src.navigation.gnss_visibility import (
    is_gnss_visible
)

from src.navigation.gnss_measurements import (
    simulate_pseudorange_set
)

from src.navigation.gnss_positioning import (
    solve_position_least_squares,
    build_pseudorange_jacobian
)

from src.faults.gnss_faults import (
    inject_pseudorange_bias
)

from src.faults.gnss_fdi import (
    global_residual_test,
    isolate_fault_leave_one_out
)


# ============================================================
# AURORA
# Expérience 006-C
# Monte Carlo du système FDI / FDIR GNSS
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE DU CUBESAT
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH + 550_000.0
)

eccentricity = 0.01

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
# 2. PARAMETRES GNSS / FDI
# ------------------------------------------------------------

measurement_time = (
    120.0 * 60.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

measurement_noise_std = 3.0

confidence_level = 0.99

number_of_runs = 500


# ------------------------------------------------------------
# 3. NIVEAUX DE FAUTE A TESTER
# ------------------------------------------------------------

fault_bias_levels = np.array([
    0.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0
])


# ------------------------------------------------------------
# 4. ETAT VRAI DU CUBESAT
# ------------------------------------------------------------

initial_position, initial_velocity = (
    keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )
)

initial_state = np.concatenate(
    (
        initial_position,
        initial_velocity
    )
)


cub_sat_solution = propagate_orbit(
    initial_state=
        initial_state,

    duration=
        measurement_time,

    number_of_points=
        2
)


true_receiver_position = (
    cub_sat_solution.y[
        0:3,
        -1
    ]
)


# ------------------------------------------------------------
# 5. CONSTELLATION GPS
# ------------------------------------------------------------

gps_constellation = (
    generate_simplified_gps_constellation()
)

gps_solutions = (
    propagate_gnss_constellation(
        constellation=
            gps_constellation,

        duration=
            measurement_time,

        number_of_points=
            2
    )
)


# ------------------------------------------------------------
# 6. SATELLITES VISIBLES
# ------------------------------------------------------------

visible_positions = []

visible_ids = []


for satellite, solution in zip(
    gps_constellation,
    gps_solutions
):

    satellite_position = (
        solution.y[
            0:3,
            -1
        ]
    )

    if is_gnss_visible(
        true_receiver_position,
        satellite_position
    ):

        visible_positions.append(
            satellite_position
        )

        visible_ids.append(
            satellite[
                "id"
            ]
        )


visible_positions = np.array(
    visible_positions
)

number_of_visible_satellites = len(
    visible_ids
)


# ------------------------------------------------------------
# 7. ESTIMATION INITIALE
# ------------------------------------------------------------

initial_position_guess = (
    true_receiver_position
    + np.array([
        20_000.0,
        -15_000.0,
        10_000.0
    ])
)


# ------------------------------------------------------------
# 8. SATELLITE FAUTE POUR CHAQUE RUN
# ------------------------------------------------------------

fault_selection_rng = (
    np.random.default_rng(
        2026
    )
)


fault_indices = (
    fault_selection_rng.integers(
        low=0,
        high=number_of_visible_satellites,
        size=number_of_runs
    )
)


# ------------------------------------------------------------
# 9. FONCTION :
# SOLUTION GNSS + TEST GLOBAL
# ------------------------------------------------------------

def solve_and_test(
    satellite_positions,
    pseudoranges
):
    """
    Résout le problème GNSS puis réalise
    le test global Chi2 sur les résidus.
    """

    solution = (
        solve_position_least_squares(
            satellite_positions=
                satellite_positions,

            pseudoranges=
                pseudoranges,

            initial_position=
                initial_position_guess,

            initial_clock_bias_range=
                0.0,

            tolerance=
                1e-4,

            max_iterations=
                20
        )
    )

    H = build_pseudorange_jacobian(
        receiver_position=
            solution[
                "position"
            ],

        satellite_positions=
            satellite_positions
    )

    rank = np.linalg.matrix_rank(
        H
    )

    degrees_of_freedom = (
        len(
            pseudoranges
        )
        - rank
    )

    test_result = (
        global_residual_test(
            residuals=
                solution[
                    "final_residuals"
                ],

            measurement_noise_std=
                measurement_noise_std,

            degrees_of_freedom=
                degrees_of_freedom,

            confidence_level=
                confidence_level
        )
    )

    return (
        solution,
        test_result
    )


# ------------------------------------------------------------
# 10. TABLEAUX DE RESULTATS
# ------------------------------------------------------------

detection_rates = []

correct_isolation_rates = []

conditional_isolation_rates = []

pre_fdir_rmse_values = []

post_fdir_rmse_values = []

mean_test_statistics = []


# ------------------------------------------------------------
# 11. BOUCLE SUR LES NIVEAUX DE BIAIS
# ------------------------------------------------------------

for fault_bias in fault_bias_levels:

    detection_count = 0

    correct_isolation_count = 0

    position_errors_before = []

    position_errors_after = []

    test_statistics = []


    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    for run_index in range(
        number_of_runs
    ):

        # ----------------------------------------------------
        # Même graine pour chaque niveau de biais :
        # comparaison appariée entre scénarios
        # ----------------------------------------------------

        measurement_seed = (
            10_000
            + run_index
        )

        measurements = (
            simulate_pseudorange_set(
                receiver_position=
                    true_receiver_position,

                satellite_positions=
                    visible_positions,

                receiver_clock_bias_seconds=
                    receiver_clock_bias_seconds,

                noise_std=
                    measurement_noise_std,

                random_seed=
                    measurement_seed
            )
        )

        pseudoranges = (
            measurements[
                "pseudoranges"
            ]
        )


        # ----------------------------------------------------
        # Injection éventuelle d'une faute
        # ----------------------------------------------------

        actual_fault_index = (
            fault_indices[
                run_index
            ]
        )

        actual_fault_id = (
            visible_ids[
                actual_fault_index
            ]
        )


        if fault_bias > 0.0:

            pseudoranges = (
                inject_pseudorange_bias(
                    pseudoranges=
                        pseudoranges,

                    satellite_index=
                        actual_fault_index,

                    bias_meters=
                        fault_bias
                )
            )


        # ----------------------------------------------------
        # Solution avant FDI
        # ----------------------------------------------------

        (
            solution,
            test_result
        ) = solve_and_test(
            satellite_positions=
                visible_positions,

            pseudoranges=
                pseudoranges
        )


        estimated_position = (
            solution[
                "position"
            ]
        )


        pre_fdir_error = np.linalg.norm(
            estimated_position
            - true_receiver_position
        )


        position_errors_before.append(
            pre_fdir_error
        )


        test_statistics.append(
            test_result[
                "statistic"
            ]
        )


        # ----------------------------------------------------
        # Par défaut :
        # si aucune détection, la navigation
        # finale reste la solution actuelle.
        # ----------------------------------------------------

        final_position_error = (
            pre_fdir_error
        )


        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        if test_result[
            "detected"
        ]:

            detection_count += 1


            # ------------------------------------------------
            # ISOLATION
            # ------------------------------------------------

            isolation_result = (
                isolate_fault_leave_one_out(
                    satellite_positions=
                        visible_positions,

                    pseudoranges=
                        pseudoranges,

                    satellite_ids=
                        visible_ids,

                    initial_position=
                        initial_position_guess,

                    initial_clock_bias_range=
                        0.0,

                    measurement_noise_std=
                        measurement_noise_std,

                    confidence_level=
                        confidence_level,

                    max_iterations=
                        20
                )
            )


            suspect_index = (
                isolation_result[
                    "suspect_index"
                ]
            )

            suspect_id = (
                isolation_result[
                    "suspect_id"
                ]
            )


            # ------------------------------------------------
            # Evaluation de l'isolation
            # ------------------------------------------------

            if (
                fault_bias > 0.0
                and suspect_id
                == actual_fault_id
            ):

                correct_isolation_count += 1


            # ------------------------------------------------
            # RECONFIGURATION
            # ------------------------------------------------

            if suspect_index is not None:

                healthy_mask = np.ones(
                    number_of_visible_satellites,
                    dtype=bool
                )

                healthy_mask[
                    suspect_index
                ] = False


                reduced_positions = (
                    visible_positions[
                        healthy_mask
                    ]
                )

                reduced_pseudoranges = (
                    pseudoranges[
                        healthy_mask
                    ]
                )


                reconfigured_solution = (
                    solve_position_least_squares(
                        satellite_positions=
                            reduced_positions,

                        pseudoranges=
                            reduced_pseudoranges,

                        initial_position=
                            initial_position_guess,

                        initial_clock_bias_range=
                            0.0,

                        tolerance=
                            1e-4,

                        max_iterations=
                            20
                    )
                )


                final_position_error = (
                    np.linalg.norm(
                        reconfigured_solution[
                            "position"
                        ]
                        - true_receiver_position
                    )
                )


        # ----------------------------------------------------
        # Erreur finale du système FDIR
        # ----------------------------------------------------

        position_errors_after.append(
            final_position_error
        )


    # --------------------------------------------------------
    # STATISTIQUES DU NIVEAU DE BIAIS
    # --------------------------------------------------------

    detection_rate = (
        100.0
        * detection_count
        / number_of_runs
    )


    if fault_bias > 0.0:

        correct_isolation_rate = (
            100.0
            * correct_isolation_count
            / number_of_runs
        )

        if detection_count > 0:

            conditional_isolation_rate = (
                100.0
                * correct_isolation_count
                / detection_count
            )

        else:

            conditional_isolation_rate = (
                0.0
            )

    else:

        correct_isolation_rate = np.nan

        conditional_isolation_rate = np.nan


    position_errors_before = np.array(
        position_errors_before
    )

    position_errors_after = np.array(
        position_errors_after
    )


    pre_fdir_rmse = np.sqrt(
        np.mean(
            position_errors_before**2
        )
    )


    post_fdir_rmse = np.sqrt(
        np.mean(
            position_errors_after**2
        )
    )


    detection_rates.append(
        detection_rate
    )

    correct_isolation_rates.append(
        correct_isolation_rate
    )

    conditional_isolation_rates.append(
        conditional_isolation_rate
    )

    pre_fdir_rmse_values.append(
        pre_fdir_rmse
    )

    post_fdir_rmse_values.append(
        post_fdir_rmse
    )

    mean_test_statistics.append(
        np.mean(
            test_statistics
        )
    )


# ------------------------------------------------------------
# 12. CONVERSION EN TABLEAUX NUMPY
# ------------------------------------------------------------

detection_rates = np.array(
    detection_rates
)

correct_isolation_rates = np.array(
    correct_isolation_rates
)

conditional_isolation_rates = np.array(
    conditional_isolation_rates
)

pre_fdir_rmse_values = np.array(
    pre_fdir_rmse_values
)

post_fdir_rmse_values = np.array(
    post_fdir_rmse_values
)

mean_test_statistics = np.array(
    mean_test_statistics
)


# ------------------------------------------------------------
# 13. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=========================================================================="
)

print(
    "AURORA — Expérience 006-C"
)

print(
    "Monte Carlo du système FDI / FDIR GNSS"
)

print(
    "=========================================================================="
)

print(
    f"Nombre de simulations par niveau : "
    f"{number_of_runs}"
)

print(
    f"Satellites visibles : "
    f"{number_of_visible_satellites}"
)

print(
    f"Bruit pseudorange : "
    f"{measurement_noise_std:.1f} m"
)

print(
    f"Niveau de confiance Chi2 : "
    f"{confidence_level * 100:.1f} %"
)

print()

print(
    "Biais | Détection | Isolation correcte | "
    "Isolation | RMSE avant | RMSE après"
)

print(
    " [m] |    [%]    |   totale [%]      | "
    "si détecté [%] |   [m]      |   [m]"
)

print(
    "--------------------------------------------------------------------------"
)


for index, fault_bias in enumerate(
    fault_bias_levels
):

    if fault_bias == 0.0:

        isolation_total_text = (
            "    N/A"
        )

        isolation_conditional_text = (
            "    N/A"
        )

    else:

        isolation_total_text = (
            f"{correct_isolation_rates[index]:7.2f}"
        )

        isolation_conditional_text = (
            f"{conditional_isolation_rates[index]:7.2f}"
        )


    print(
        f"{fault_bias:5.1f} | "
        f"{detection_rates[index]:8.2f} | "
        f"{isolation_total_text:>17} | "
        f"{isolation_conditional_text:>15} | "
        f"{pre_fdir_rmse_values[index]:10.3f} | "
        f"{post_fdir_rmse_values[index]:10.3f}"
    )


print(
    "=========================================================================="
)


print()

print(
    "Pour le cas biais = 0 m, le taux de "
    "détection correspond au taux de fausse alarme."
)

print(
    f"Fausse alarme observée : "
    f"{detection_rates[0]:.2f} %"
)

print(
    f"Fausse alarme théorique approximative : "
    f"{(1.0 - confidence_level) * 100.0:.2f} %"
)


# ------------------------------------------------------------
# 14. FIGURE - PROBABILITE DE DETECTION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    fault_bias_levels,
    detection_rates,
    marker="o",
    label="Taux de détection"
)

plt.axhline(
    (1.0 - confidence_level) * 100.0,
    linestyle="--",
    label="Fausse alarme nominale attendue"
)

plt.xlabel(
    "Biais de pseudorange injecté [m]"
)

plt.ylabel(
    "Détection [%]"
)

plt.title(
    "AURORA — Probabilité de détection GNSS"
)

plt.ylim(
    0,
    105
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. FIGURE - PROBABILITE D'ISOLATION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    fault_bias_levels[1:],
    correct_isolation_rates[1:],
    marker="o",
    label="Isolation correcte totale"
)

plt.plot(
    fault_bias_levels[1:],
    conditional_isolation_rates[1:],
    marker="o",
    label="Isolation correcte si détection"
)

plt.xlabel(
    "Biais de pseudorange injecté [m]"
)

plt.ylabel(
    "Isolation correcte [%]"
)

plt.title(
    "AURORA — Performance d'isolation du satellite fautif"
)

plt.ylim(
    0,
    105
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FIGURE - PERFORMANCE DE NAVIGATION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    fault_bias_levels,
    pre_fdir_rmse_values,
    marker="o",
    label="Avant FDI"
)

plt.plot(
    fault_bias_levels,
    post_fdir_rmse_values,
    marker="o",
    label="Après logique FDIR"
)

plt.xlabel(
    "Biais de pseudorange injecté [m]"
)

plt.ylabel(
    "RMSE de position 3D [m]"
)

plt.title(
    "AURORA — Bénéfice de la reconfiguration GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()