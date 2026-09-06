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
    C_LIGHT,
    simulate_pseudorange_set
)

from src.navigation.gnss_positioning import (
    solve_position_least_squares
)

from src.navigation.gnss_geometry import (
    compute_dop
)


# ============================================================
# AURORA
# Expérience 005-D
# Performance GNSS au cours de l'orbite
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE DU CUBESAT
# ------------------------------------------------------------

cub_sat_semi_major_axis = (
    R_EARTH + 550_000.0
)

cub_sat_eccentricity = 0.01

cub_sat_inclination = np.deg2rad(
    97.6
)

cub_sat_raan = np.deg2rad(
    40.0
)

cub_sat_argument_of_periapsis = (
    np.deg2rad(30.0)
)

cub_sat_true_anomaly = (
    np.deg2rad(25.0)
)


# ------------------------------------------------------------
# 2. PARAMETRES GNSS
# ------------------------------------------------------------

receiver_clock_bias_seconds = (
    100.0e-6
)

pseudorange_noise_std = 3.0

true_clock_bias_range = (
    C_LIGHT
    * receiver_clock_bias_seconds
)


# ------------------------------------------------------------
# 3. DUREE DE SIMULATION
# ------------------------------------------------------------

cub_sat_period = orbital_period(
    cub_sat_semi_major_axis
)

simulation_duration = (
    3.0
    * cub_sat_period
)

number_of_epochs = 600


# ------------------------------------------------------------
# 4. ETAT INITIAL DU CUBESAT
# ------------------------------------------------------------

initial_position, initial_velocity = (
    keplerian_to_cartesian(
        cub_sat_semi_major_axis,
        cub_sat_eccentricity,
        cub_sat_inclination,
        cub_sat_raan,
        cub_sat_argument_of_periapsis,
        cub_sat_true_anomaly
    )
)

cub_sat_initial_state = np.concatenate(
    (
        initial_position,
        initial_velocity
    )
)


# ------------------------------------------------------------
# 5. PROPAGATION DU CUBESAT
# ------------------------------------------------------------

cub_sat_solution = propagate_orbit(
    initial_state=cub_sat_initial_state,
    duration=simulation_duration,
    number_of_points=number_of_epochs
)


# ------------------------------------------------------------
# 6. CONSTELLATION GPS
# ------------------------------------------------------------

gps_constellation = (
    generate_simplified_gps_constellation()
)

gps_solutions = (
    propagate_gnss_constellation(
        constellation=gps_constellation,
        duration=simulation_duration,
        number_of_points=number_of_epochs
    )
)


# ------------------------------------------------------------
# 7. TABLEAUX DE RESULTATS
# ------------------------------------------------------------

visible_satellite_count = np.zeros(
    number_of_epochs,
    dtype=int
)

pdop_values = np.full(
    number_of_epochs,
    np.nan
)

tdop_values = np.full(
    number_of_epochs,
    np.nan
)

position_errors = np.full(
    number_of_epochs,
    np.nan
)

clock_bias_errors = np.full(
    number_of_epochs,
    np.nan
)

residual_rms_values = np.full(
    number_of_epochs,
    np.nan
)

iteration_counts = np.zeros(
    number_of_epochs,
    dtype=int
)

convergence_flags = np.zeros(
    number_of_epochs,
    dtype=bool
)


# ------------------------------------------------------------
# 8. ERREUR INITIALE DU SOLVEUR
# ------------------------------------------------------------

initial_position_offset = np.array([
    20_000.0,
    -15_000.0,
    10_000.0
])


# ------------------------------------------------------------
# 9. BOUCLE TEMPORELLE GNSS
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
    # Recherche des satellites visibles
    # --------------------------------------------------------

    visible_positions = []

    for gps_solution in gps_solutions:

        satellite_position = (
            gps_solution.y[
                0:3,
                time_index
            ]
        )

        if is_gnss_visible(
            true_receiver_position,
            satellite_position
        ):

            visible_positions.append(
                satellite_position
            )

    visible_satellite_count[
        time_index
    ] = len(
        visible_positions
    )

    # --------------------------------------------------------
    # Pas de solution si moins de 4 satellites
    # --------------------------------------------------------

    if len(
        visible_positions
    ) < 4:

        continue

    visible_positions = np.array(
        visible_positions
    )

    # --------------------------------------------------------
    # Calcul DOP
    # --------------------------------------------------------

    dop_results = compute_dop(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            visible_positions
    )

    pdop_values[
        time_index
    ] = dop_results[
        "pdop"
    ]

    tdop_values[
        time_index
    ] = dop_results[
        "tdop"
    ]

    # --------------------------------------------------------
    # Simulation des pseudoranges
    # --------------------------------------------------------

    measurements = simulate_pseudorange_set(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            visible_positions,

        receiver_clock_bias_seconds=
            receiver_clock_bias_seconds,

        noise_std=
            pseudorange_noise_std,

        random_seed=
            5000 + time_index
    )

    pseudoranges = measurements[
        "pseudoranges"
    ]

    # --------------------------------------------------------
    # Estimation initiale
    # --------------------------------------------------------

    initial_position_guess = (
        true_receiver_position
        + initial_position_offset
    )

    # --------------------------------------------------------
    # Solution GNSS
    # --------------------------------------------------------

    solution = solve_position_least_squares(
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

    convergence_flags[
        time_index
    ] = solution[
        "converged"
    ]

    iteration_counts[
        time_index
    ] = solution[
        "iterations"
    ]

    if not solution[
        "converged"
    ]:

        continue

    # --------------------------------------------------------
    # Erreur de position
    # --------------------------------------------------------

    estimated_position = solution[
        "position"
    ]

    position_errors[
        time_index
    ] = np.linalg.norm(
        estimated_position
        - true_receiver_position
    )

    # --------------------------------------------------------
    # Erreur d'horloge
    # --------------------------------------------------------

    clock_bias_errors[
        time_index
    ] = (
        solution[
            "clock_bias_range"
        ]
        - true_clock_bias_range
    )

    # --------------------------------------------------------
    # Résidus
    # --------------------------------------------------------

    residual_rms_values[
        time_index
    ] = solution[
        "final_residual_rms"
    ]


# ------------------------------------------------------------
# 10. DONNEES VALIDES
# ------------------------------------------------------------

valid_mask = (
    np.isfinite(
        position_errors
    )
    &
    np.isfinite(
        pdop_values
    )
)

valid_position_errors = (
    position_errors[
        valid_mask
    ]
)

valid_pdop = (
    pdop_values[
        valid_mask
    ]
)

valid_tdop = (
    tdop_values[
        valid_mask
    ]
)

valid_clock_bias_errors = (
    clock_bias_errors[
        valid_mask
    ]
)


# ------------------------------------------------------------
# 11. STATISTIQUES DE POSITION
# ------------------------------------------------------------

position_rmse = np.sqrt(
    np.mean(
        valid_position_errors**2
    )
)

mean_position_error = np.mean(
    valid_position_errors
)

median_position_error = np.median(
    valid_position_errors
)

position_error_95 = np.percentile(
    valid_position_errors,
    95
)

maximum_position_error = np.max(
    valid_position_errors
)


# ------------------------------------------------------------
# 12. PREDICTION THEORIQUE SUR TOUTE L'ORBITE
# ------------------------------------------------------------

predicted_orbit_position_rmse = (
    pseudorange_noise_std
    * np.sqrt(
        np.mean(
            valid_pdop**2
        )
    )
)


predicted_orbit_clock_rmse = (
    pseudorange_noise_std
    * np.sqrt(
        np.mean(
            valid_tdop**2
        )
    )
)


# ------------------------------------------------------------
# 13. RMSE HORLOGE
# ------------------------------------------------------------

clock_bias_rmse = np.sqrt(
    np.mean(
        valid_clock_bias_errors**2
    )
)


# ------------------------------------------------------------
# 14. DISPONIBILITE
# ------------------------------------------------------------

four_satellite_availability = (
    np.mean(
        visible_satellite_count >= 4
    )
    * 100.0
)

solution_availability = (
    np.mean(
        np.isfinite(
            position_errors
        )
    )
    * 100.0
)

convergence_rate = (
    np.mean(
        convergence_flags[
            visible_satellite_count >= 4
        ]
    )
    * 100.0
)


# ------------------------------------------------------------
# 15. CORRELATION PDOP / ERREUR
# ------------------------------------------------------------

if len(
    valid_position_errors
) > 1:

    pdop_error_correlation = (
        np.corrcoef(
            valid_pdop,
            valid_position_errors
        )[0, 1]
    )

else:

    pdop_error_correlation = np.nan


# ------------------------------------------------------------
# 16. RESULTATS NUMERIQUES
# ------------------------------------------------------------

print(
    "\n"
    "======================================================"
)

print(
    "AURORA — Expérience 005-D"
)

print(
    "Performance GNSS au cours de l'orbite"
)

print(
    "======================================================"
)

print(
    f"Durée simulée : "
    f"{simulation_duration / 60.0:.2f} min"
)

print(
    f"Nombre d'époques GNSS : "
    f"{number_of_epochs}"
)

print()

print(
    "----- VISIBILITE -----"
)

print(
    f"Minimum visible : "
    f"{np.min(visible_satellite_count)}"
)

print(
    f"Maximum visible : "
    f"{np.max(visible_satellite_count)}"
)

print(
    f"Moyenne visible : "
    f"{np.mean(visible_satellite_count):.2f}"
)

print(
    f"Disponibilité >= 4 satellites : "
    f"{four_satellite_availability:.2f} %"
)

print()

print(
    "----- PDOP -----"
)

print(
    f"PDOP minimum : "
    f"{np.min(valid_pdop):.3f}"
)

print(
    f"PDOP moyen : "
    f"{np.mean(valid_pdop):.3f}"
)

print(
    f"PDOP maximum : "
    f"{np.max(valid_pdop):.3f}"
)

print()

print(
    "----- POSITION -----"
)

print(
    f"RMSE position théorique orbital : "
    f"{predicted_orbit_position_rmse:.3f} m"
)

print(
    f"RMSE position simulé : "
    f"{position_rmse:.3f} m"
)

print(
    f"Ratio simulation / théorie : "
    f"{position_rmse / predicted_orbit_position_rmse:.3f}"
)

print()

print(
    f"Erreur moyenne : "
    f"{mean_position_error:.3f} m"
)

print(
    f"Erreur médiane : "
    f"{median_position_error:.3f} m"
)

print(
    f"95e percentile : "
    f"{position_error_95:.3f} m"
)

print(
    f"Erreur maximale : "
    f"{maximum_position_error:.3f} m"
)

print()

print(
    "----- HORLOGE -----"
)

print(
    f"RMSE horloge théorique : "
    f"{predicted_orbit_clock_rmse:.3f} m"
)

print(
    f"RMSE horloge simulé : "
    f"{clock_bias_rmse:.3f} m"
)

print()

print(
    "----- SOLVEUR -----"
)

print(
    f"Taux de convergence : "
    f"{convergence_rate:.2f} %"
)

print(
    f"Disponibilité solution GNSS : "
    f"{solution_availability:.2f} %"
)

print(
    f"Nombre moyen d'itérations : "
    f"{np.mean(iteration_counts[valid_mask]):.2f}"
)

print()

print(
    f"Corrélation PDOP / erreur : "
    f"{pdop_error_correlation:.3f}"
)

print(
    "======================================================"
)


# ------------------------------------------------------------
# 17. TEMPS EN MINUTES
# ------------------------------------------------------------

time_minutes = (
    cub_sat_solution.t
    / 60.0
)


# ------------------------------------------------------------
# 18. FIGURE ERREUR DE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    time_minutes,
    position_errors,
    label="Erreur GNSS 3D"
)

plt.plot(
    time_minutes,
    pdop_values
    * pseudorange_noise_std,
    linestyle="--",
    label="Echelle PDOP × sigma"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Erreur / échelle théorique [m]"
)

plt.title(
    "AURORA — Erreur de position GNSS au cours de l'orbite"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FIGURE PDOP
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    time_minutes,
    pdop_values
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "PDOP [-]"
)

plt.title(
    "AURORA — Evolution du PDOP"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. FIGURE SATELLITES VISIBLES
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    time_minutes,
    visible_satellite_count
)

plt.axhline(
    4,
    linestyle="--",
    label="Minimum de 4 satellites"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Satellites GPS visibles"
)

plt.title(
    "AURORA — Disponibilité géométrique GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. FIGURE PDOP VS ERREUR
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    valid_pdop,
    valid_position_errors,
    s=15,
    alpha=0.6
)

plt.xlabel(
    "PDOP [-]"
)

plt.ylabel(
    "Erreur GNSS 3D [m]"
)

plt.title(
    "AURORA — Relation PDOP / erreur de position"
)

plt.grid(True)

plt.tight_layout()

plt.show()