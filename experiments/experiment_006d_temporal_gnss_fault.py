import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH,
    orbital_period,
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
# Expérience 006-D
# Faute GNSS temporelle et FDIR
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
# 2. PARAMETRES GNSS
# ------------------------------------------------------------

receiver_clock_bias_seconds = (
    100.0e-6
)

measurement_noise_std = 3.0

confidence_level = 0.99


# ------------------------------------------------------------
# 3. SCENARIO DE FAUTE
# ------------------------------------------------------------

fault_bias_meters = 50.0

fault_start_minutes = 100.0

fault_end_minutes = 130.0


# ------------------------------------------------------------
# 4. DUREE DE SIMULATION
# ------------------------------------------------------------

cub_sat_period = orbital_period(
    semi_major_axis
)

simulation_duration = (
    2.0 * cub_sat_period
)

number_of_epochs = 300


# ------------------------------------------------------------
# 5. ETAT INITIAL
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


# ------------------------------------------------------------
# 6. PROPAGATION DU CUBESAT
# ------------------------------------------------------------

cub_sat_solution = propagate_orbit(
    initial_state=
        initial_state,

    duration=
        simulation_duration,

    number_of_points=
        number_of_epochs
)


# ------------------------------------------------------------
# 7. CONSTELLATION GPS
# ------------------------------------------------------------

gps_constellation = (
    generate_simplified_gps_constellation()
)

gps_solutions = (
    propagate_gnss_constellation(
        constellation=
            gps_constellation,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs
    )
)


# ------------------------------------------------------------
# 8. TEMPS
# ------------------------------------------------------------

time_seconds = (
    cub_sat_solution.t
)

time_minutes = (
    time_seconds / 60.0
)

fault_window_mask = (
    (time_minutes >= fault_start_minutes)
    &
    (time_minutes <= fault_end_minutes)
)


# ------------------------------------------------------------
# 9. VISIBILITE DE CHAQUE GPS
# ------------------------------------------------------------

visibility_matrix = np.zeros(
    (
        len(gps_constellation),
        number_of_epochs
    ),
    dtype=bool
)


for time_index in range(
    number_of_epochs
):

    receiver_position = (
        cub_sat_solution.y[
            0:3,
            time_index
        ]
    )

    for gps_index, gps_solution in enumerate(
        gps_solutions
    ):

        satellite_position = (
            gps_solution.y[
                0:3,
                time_index
            ]
        )

        visibility_matrix[
            gps_index,
            time_index
        ] = is_gnss_visible(
            receiver_position,
            satellite_position
        )


# ------------------------------------------------------------
# 10. CHOIX DU SATELLITE A FAUTER
# ------------------------------------------------------------

fault_epoch_indices = np.where(
    fault_window_mask
)[0]


visibility_during_fault = np.sum(
    visibility_matrix[
        :,
        fault_epoch_indices
    ],
    axis=1
)


target_gps_index = np.argmax(
    visibility_during_fault
)

target_gps_id = (
    gps_constellation[
        target_gps_index
    ][
        "id"
    ]
)


# ------------------------------------------------------------
# 11. TABLEAUX DE RESULTATS
# ------------------------------------------------------------

number_visible = np.zeros(
    number_of_epochs,
    dtype=int
)

position_error_before_fdir = np.full(
    number_of_epochs,
    np.nan
)

position_error_after_fdir = np.full(
    number_of_epochs,
    np.nan
)

test_statistics = np.full(
    number_of_epochs,
    np.nan
)

test_thresholds = np.full(
    number_of_epochs,
    np.nan
)

detection_flags = np.zeros(
    number_of_epochs,
    dtype=bool
)

correct_isolation_flags = np.zeros(
    number_of_epochs,
    dtype=bool
)

fault_actually_present = np.zeros(
    number_of_epochs,
    dtype=bool
)


# ------------------------------------------------------------
# 12. OFFSET INITIAL DU SOLVEUR
# ------------------------------------------------------------

initial_position_offset = np.array([
    20_000.0,
    -15_000.0,
    10_000.0
])


# ------------------------------------------------------------
# 13. BOUCLE TEMPORELLE
# ------------------------------------------------------------

for time_index in range(
    number_of_epochs
):

    true_receiver_position = (
        cub_sat_solution.y[
            0:3,
            time_index
        ]
    )


    # --------------------------------------------------------
    # Satellites visibles
    # --------------------------------------------------------

    visible_positions = []

    visible_ids = []


    for gps_index, gps_solution in enumerate(
        gps_solutions
    ):

        if not visibility_matrix[
            gps_index,
            time_index
        ]:
            continue

        visible_positions.append(
            gps_solution.y[
                0:3,
                time_index
            ]
        )

        visible_ids.append(
            gps_constellation[
                gps_index
            ][
                "id"
            ]
        )


    visible_positions = np.array(
        visible_positions
    )

    number_visible[
        time_index
    ] = len(
        visible_ids
    )


    if len(
        visible_ids
    ) < 4:
        continue


    # --------------------------------------------------------
    # Pseudoranges nominales
    # --------------------------------------------------------

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
                20_000
                + time_index
        )
    )


    pseudoranges = (
        measurements[
            "pseudoranges"
        ]
    )


    # --------------------------------------------------------
    # Injection de la faute
    # --------------------------------------------------------

    target_visible = (
        target_gps_id
        in visible_ids
    )

    fault_active = (
        fault_window_mask[
            time_index
        ]
        and target_visible
    )


    if fault_active:

        local_fault_index = (
            visible_ids.index(
                target_gps_id
            )
        )

        pseudoranges = (
            inject_pseudorange_bias(
                pseudoranges=
                    pseudoranges,

                satellite_index=
                    local_fault_index,

                bias_meters=
                    fault_bias_meters
            )
        )

        fault_actually_present[
            time_index
        ] = True


    # --------------------------------------------------------
    # Estimation initiale
    # --------------------------------------------------------

    initial_position_guess = (
        true_receiver_position
        + initial_position_offset
    )


    # --------------------------------------------------------
    # SOLUTION AVANT FDIR
    # --------------------------------------------------------

    solution = (
        solve_position_least_squares(
            satellite_positions=
                visible_positions,

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


    position_error_before_fdir[
        time_index
    ] = np.linalg.norm(
        solution[
            "position"
        ]
        - true_receiver_position
    )


    # --------------------------------------------------------
    # TEST GLOBAL
    # --------------------------------------------------------

    H = build_pseudorange_jacobian(
        receiver_position=
            solution[
                "position"
            ],

        satellite_positions=
            visible_positions
    )


    rank = np.linalg.matrix_rank(
        H
    )


    degrees_of_freedom = (
        len(
            visible_ids
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


    test_statistics[
        time_index
    ] = test_result[
        "statistic"
    ]

    test_thresholds[
        time_index
    ] = test_result[
        "threshold"
    ]

    detection_flags[
        time_index
    ] = test_result[
        "detected"
    ]


    # --------------------------------------------------------
    # Par défaut :
    # pas de reconfiguration
    # --------------------------------------------------------

    final_position = (
        solution[
            "position"
        ].copy()
    )


    # --------------------------------------------------------
    # FDI / RECONFIGURATION
    # --------------------------------------------------------

    if test_result[
        "detected"
    ]:

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


        if (
            fault_active
            and suspect_id
            == target_gps_id
        ):

            correct_isolation_flags[
                time_index
            ] = True


        if suspect_index is not None:

            healthy_mask = np.ones(
                len(
                    visible_ids
                ),
                dtype=bool
            )

            healthy_mask[
                suspect_index
            ] = False


            reconfigured_solution = (
                solve_position_least_squares(
                    satellite_positions=
                        visible_positions[
                            healthy_mask
                        ],

                    pseudoranges=
                        pseudoranges[
                            healthy_mask
                        ],

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


            final_position = (
                reconfigured_solution[
                    "position"
                ]
            )


    # --------------------------------------------------------
    # ERREUR APRES FDIR
    # --------------------------------------------------------

    position_error_after_fdir[
        time_index
    ] = np.linalg.norm(
        final_position
        - true_receiver_position
    )


# ------------------------------------------------------------
# 14. MASQUES STATISTIQUES
# ------------------------------------------------------------

fault_active_mask = (
    fault_actually_present
)

healthy_mask = (
    ~fault_window_mask
)

valid_before = np.isfinite(
    position_error_before_fdir
)

valid_after = np.isfinite(
    position_error_after_fdir
)


# ------------------------------------------------------------
# 15. DETECTION PENDANT LA FAUTE
# ------------------------------------------------------------

number_fault_epochs = np.sum(
    fault_active_mask
)


if number_fault_epochs > 0:

    detection_rate_during_fault = (
        100.0
        * np.sum(
            detection_flags
            & fault_active_mask
        )
        / number_fault_epochs
    )

    correct_isolation_rate = (
        100.0
        * np.sum(
            correct_isolation_flags
            & fault_active_mask
        )
        / number_fault_epochs
    )

else:

    detection_rate_during_fault = np.nan

    correct_isolation_rate = np.nan


# ------------------------------------------------------------
# 16. FAUSSES ALARMES HORS FAUTE
# ------------------------------------------------------------

healthy_valid_mask = (
    healthy_mask
    & valid_before
)


false_alarm_rate = (
    100.0
    * np.sum(
        detection_flags
        & healthy_valid_mask
    )
    / np.sum(
        healthy_valid_mask
    )
)


# ------------------------------------------------------------
# 17. RMSE PENDANT LA FAUTE
# ------------------------------------------------------------

fault_before_values = (
    position_error_before_fdir[
        fault_active_mask
    ]
)

fault_after_values = (
    position_error_after_fdir[
        fault_active_mask
    ]
)


rmse_before_fault = np.sqrt(
    np.mean(
        fault_before_values**2
    )
)


rmse_after_fault = np.sqrt(
    np.mean(
        fault_after_values**2
    )
)


# ------------------------------------------------------------
# 18. RMSE HORS FAUTE
# ------------------------------------------------------------

healthy_before_values = (
    position_error_before_fdir[
        healthy_valid_mask
    ]
)


healthy_after_values = (
    position_error_after_fdir[
        healthy_valid_mask
    ]
)


healthy_rmse_before = np.sqrt(
    np.mean(
        healthy_before_values**2
    )
)


healthy_rmse_after = np.sqrt(
    np.mean(
        healthy_after_values**2
    )
)


# ------------------------------------------------------------
# 19. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================"
)

print(
    "AURORA — Expérience 006-D"
)

print(
    "Faute GNSS temporelle et reconfiguration FDIR"
)

print(
    "============================================================"
)

print(
    f"Satellite fauté : "
    f"GPS {target_gps_id:02d}"
)

print(
    f"Biais injecté : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"Fenêtre de faute demandée : "
    f"{fault_start_minutes:.1f} "
    f"à {fault_end_minutes:.1f} min"
)

print(
    f"Epoques où la faute est réellement active : "
    f"{number_fault_epochs}"
)

print()

print(
    "----- DETECTION / ISOLATION -----"
)

print(
    f"Taux de détection pendant la faute : "
    f"{detection_rate_during_fault:.2f} %"
)

print(
    f"Taux d'isolation correcte : "
    f"{correct_isolation_rate:.2f} %"
)

print(
    f"Taux de fausse alarme hors faute : "
    f"{false_alarm_rate:.2f} %"
)

print()

print(
    "----- PERFORMANCE PENDANT LA FAUTE -----"
)

print(
    f"RMSE avant FDIR : "
    f"{rmse_before_fault:.3f} m"
)

print(
    f"RMSE après FDIR : "
    f"{rmse_after_fault:.3f} m"
)

print()

print(
    "----- PERFORMANCE HORS FAUTE -----"
)

print(
    f"RMSE sans reconfiguration : "
    f"{healthy_rmse_before:.3f} m"
)

print(
    f"RMSE avec logique FDIR : "
    f"{healthy_rmse_after:.3f} m"
)

print(
    "============================================================"
)


# ------------------------------------------------------------
# 20. FIGURE ERREUR DE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    position_error_before_fdir,
    label="Avant FDIR"
)

plt.plot(
    time_minutes,
    position_error_after_fdir,
    label="Après FDIR"
)

plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15,
    label="Fenêtre de faute"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur de position 3D [m]"
)

plt.title(
    "AURORA — Navigation avant / après FDIR"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. FIGURE TEST CHI2
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_minutes,
    test_statistics,
    label="Statistique Chi2"
)

plt.plot(
    time_minutes,
    test_thresholds,
    linestyle="--",
    label="Seuil de détection"
)

plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Statistique du test [-]"
)

plt.title(
    "AURORA — Détection temporelle de l'anomalie GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. FIGURE DETECTION BINAIRE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 4)
)

plt.step(
    time_minutes,
    detection_flags.astype(
        int
    ),
    where="mid",
    label="Détection"
)

plt.step(
    time_minutes,
    fault_actually_present.astype(
        int
    ),
    where="mid",
    label="Faute réelle"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Etat binaire"
)

plt.yticks([
    0,
    1
])

plt.title(
    "AURORA — Faute réelle vs décision FDI"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()