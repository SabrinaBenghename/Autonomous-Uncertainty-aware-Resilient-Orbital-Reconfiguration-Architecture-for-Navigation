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
# Expérience 005-C
# Monte Carlo du positionnement GNSS
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

measurement_time = (
    120.0 * 60.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

pseudorange_noise_std = 3.0

number_of_monte_carlo_runs = 1000


# ------------------------------------------------------------
# 3. ETAT VRAI DU CUBESAT
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

cub_sat_solution = propagate_orbit(
    initial_state=cub_sat_initial_state,
    duration=measurement_time,
    number_of_points=2
)

true_receiver_position = (
    cub_sat_solution.y[
        0:3,
        -1
    ]
)


# ------------------------------------------------------------
# 4. CONSTELLATION GPS
# ------------------------------------------------------------

gps_constellation = (
    generate_simplified_gps_constellation()
)

gps_solutions = (
    propagate_gnss_constellation(
        constellation=gps_constellation,
        duration=measurement_time,
        number_of_points=2
    )
)


# ------------------------------------------------------------
# 5. SATELLITES VISIBLES
# ------------------------------------------------------------

visible_satellite_positions = []

visible_satellite_ids = []


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

        visible_satellite_positions.append(
            satellite_position
        )

        visible_satellite_ids.append(
            satellite["id"]
        )


visible_satellite_positions = np.array(
    visible_satellite_positions
)


# ------------------------------------------------------------
# 6. DOP THEORIQUE
# ------------------------------------------------------------

dop_results = compute_dop(
    receiver_position=
        true_receiver_position,

    satellite_positions=
        visible_satellite_positions
)

pdop = dop_results[
    "pdop"
]

tdop = dop_results[
    "tdop"
]

gdop = dop_results[
    "gdop"
]


# ------------------------------------------------------------
# 7. PREDICTIONS THEORIQUES
# ------------------------------------------------------------

predicted_position_rmse = (
    pdop
    * pseudorange_noise_std
)

predicted_clock_bias_rmse = (
    tdop
    * pseudorange_noise_std
)

number_of_measurements = len(
    visible_satellite_positions
)

number_of_estimated_parameters = 4

predicted_residual_rms = (
    pseudorange_noise_std
    * np.sqrt(
        (
            number_of_measurements
            - number_of_estimated_parameters
        )
        /
        number_of_measurements
    )
)


# ------------------------------------------------------------
# 8. BIAIS D'HORLOGE VRAI
# ------------------------------------------------------------

true_clock_bias_range = (
    C_LIGHT
    * receiver_clock_bias_seconds
)


# ------------------------------------------------------------
# 9. ESTIMATION INITIALE
# ------------------------------------------------------------

initial_position_offset = np.array([
    20_000.0,
    -15_000.0,
    10_000.0
])

initial_position_guess = (
    true_receiver_position
    + initial_position_offset
)

initial_clock_bias_range_guess = 0.0


# ------------------------------------------------------------
# 10. TABLEAUX MONTE CARLO
# ------------------------------------------------------------

position_errors = np.zeros(
    number_of_monte_carlo_runs
)

position_error_vectors = np.zeros(
    (
        number_of_monte_carlo_runs,
        3
    )
)

clock_bias_errors = np.zeros(
    number_of_monte_carlo_runs
)

residual_rms_values = np.zeros(
    number_of_monte_carlo_runs
)

iteration_counts = np.zeros(
    number_of_monte_carlo_runs,
    dtype=int
)

convergence_flags = np.zeros(
    number_of_monte_carlo_runs,
    dtype=bool
)


# ------------------------------------------------------------
# 11. BOUCLE MONTE CARLO
# ------------------------------------------------------------

for run_index in range(
    number_of_monte_carlo_runs
):

    measurements = simulate_pseudorange_set(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            visible_satellite_positions,

        receiver_clock_bias_seconds=
            receiver_clock_bias_seconds,

        noise_std=
            pseudorange_noise_std,

        random_seed=
            1000 + run_index
    )

    pseudoranges = measurements[
        "pseudoranges"
    ]

    solution = solve_position_least_squares(
        satellite_positions=
            visible_satellite_positions,

        pseudoranges=
            pseudoranges,

        initial_position=
            initial_position_guess,

        initial_clock_bias_range=
            initial_clock_bias_range_guess,

        tolerance=
            1e-4,

        max_iterations=
            20
    )

    estimated_position = solution[
        "position"
    ]

    estimated_clock_bias_range = solution[
        "clock_bias_range"
    ]

    position_error_vector = (
        estimated_position
        - true_receiver_position
    )

    position_error = np.linalg.norm(
        position_error_vector
    )

    clock_bias_error = (
        estimated_clock_bias_range
        - true_clock_bias_range
    )

    position_error_vectors[
        run_index
    ] = position_error_vector

    position_errors[
        run_index
    ] = position_error

    clock_bias_errors[
        run_index
    ] = clock_bias_error

    residual_rms_values[
        run_index
    ] = solution[
        "final_residual_rms"
    ]

    iteration_counts[
        run_index
    ] = solution[
        "iterations"
    ]

    convergence_flags[
        run_index
    ] = solution[
        "converged"
    ]


# ------------------------------------------------------------
# 12. STATISTIQUES POSITION
# ------------------------------------------------------------

position_rmse = np.sqrt(
    np.mean(
        position_errors**2
    )
)

mean_position_error = np.mean(
    position_errors
)

median_position_error = np.median(
    position_errors
)

position_error_95 = np.percentile(
    position_errors,
    95
)

maximum_position_error = np.max(
    position_errors
)


# ------------------------------------------------------------
# 13. BIAIS CARTESIEN MOYEN
# ------------------------------------------------------------

mean_position_error_vector = np.mean(
    position_error_vectors,
    axis=0
)

mean_position_bias_norm = np.linalg.norm(
    mean_position_error_vector
)


# ------------------------------------------------------------
# 14. STATISTIQUES HORLOGE
# ------------------------------------------------------------

clock_bias_rmse = np.sqrt(
    np.mean(
        clock_bias_errors**2
    )
)

clock_bias_mean_error = np.mean(
    clock_bias_errors
)


# ------------------------------------------------------------
# 15. STATISTIQUES RESIDUS
# ------------------------------------------------------------

mean_residual_rms = np.mean(
    residual_rms_values
)


# ------------------------------------------------------------
# 16. CONVERGENCE
# ------------------------------------------------------------

convergence_rate = (
    np.mean(
        convergence_flags
    )
    * 100.0
)

mean_iterations = np.mean(
    iteration_counts
)


# ------------------------------------------------------------
# 17. RATIOS EXPERIENCE / THEORIE
# ------------------------------------------------------------

position_rmse_ratio = (
    position_rmse
    / predicted_position_rmse
)

clock_rmse_ratio = (
    clock_bias_rmse
    / predicted_clock_bias_rmse
)

residual_rms_ratio = (
    mean_residual_rms
    / predicted_residual_rms
)


# ------------------------------------------------------------
# 18. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "======================================================"
)

print(
    "AURORA — Expérience 005-C"
)

print(
    "Monte Carlo du positionnement GNSS"
)

print(
    "======================================================"
)

print(
    f"Nombre de simulations : "
    f"{number_of_monte_carlo_runs}"
)

print(
    f"Satellites visibles : "
    f"{number_of_measurements}"
)

print()

print(
    f"PDOP : "
    f"{pdop:.3f}"
)

print(
    f"TDOP : "
    f"{tdop:.3f}"
)

print(
    f"GDOP : "
    f"{gdop:.3f}"
)

print()

print(
    "----- POSITION -----"
)

print(
    f"RMSE position théorique "
    f"(PDOP * sigma) : "
    f"{predicted_position_rmse:.3f} m"
)

print(
    f"RMSE position Monte Carlo : "
    f"{position_rmse:.3f} m"
)

print(
    f"Ratio expérience / théorie : "
    f"{position_rmse_ratio:.3f}"
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
    f"Norme du biais cartésien moyen : "
    f"{mean_position_bias_norm:.3f} m"
)

print()

print(
    "----- HORLOGE -----"
)

print(
    f"RMSE horloge théorique : "
    f"{predicted_clock_bias_rmse:.3f} m"
)

print(
    f"RMSE horloge Monte Carlo : "
    f"{clock_bias_rmse:.3f} m"
)

print(
    f"Ratio expérience / théorie : "
    f"{clock_rmse_ratio:.3f}"
)

print(
    f"Erreur moyenne horloge : "
    f"{clock_bias_mean_error:.3f} m"
)

print()

print(
    "----- RESIDUS -----"
)

print(
    f"RMS résiduel théorique : "
    f"{predicted_residual_rms:.3f} m"
)

print(
    f"RMS résiduel moyen Monte Carlo : "
    f"{mean_residual_rms:.3f} m"
)

print(
    f"Ratio expérience / théorie : "
    f"{residual_rms_ratio:.3f}"
)

print()

print(
    "----- CONVERGENCE -----"
)

print(
    f"Taux de convergence : "
    f"{convergence_rate:.2f} %"
)

print(
    f"Nombre moyen d'itérations : "
    f"{mean_iterations:.2f}"
)

print(
    "======================================================"
)


# ------------------------------------------------------------
# 19. HISTOGRAMME ERREUR DE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.hist(
    position_errors,
    bins=40
)

plt.axvline(
    predicted_position_rmse,
    linestyle="--",
    label="RMSE théorique"
)

plt.axvline(
    position_rmse,
    linestyle=":",
    label="RMSE Monte Carlo"
)

plt.xlabel(
    "Erreur 3D de position [m]"
)

plt.ylabel(
    "Nombre de simulations"
)

plt.title(
    "AURORA — Distribution de l'erreur GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. HISTOGRAMME ERREUR HORLOGE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.hist(
    clock_bias_errors,
    bins=40
)

plt.axvline(
    0.0,
    linestyle="--"
)

plt.xlabel(
    "Erreur du biais d'horloge [m]"
)

plt.ylabel(
    "Nombre de simulations"
)

plt.title(
    "AURORA — Erreur d'estimation du biais d'horloge"
)

plt.grid(True)

plt.tight_layout()

plt.show()