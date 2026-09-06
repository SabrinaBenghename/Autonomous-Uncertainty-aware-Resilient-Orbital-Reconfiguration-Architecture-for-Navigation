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
# Expérience 005-B
# Positionnement GNSS par moindres carrés itératifs
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
# 2. PARAMETRES DE MESURE
# ------------------------------------------------------------

measurement_time = (
    120.0 * 60.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

pseudorange_noise_std = 3.0

random_seed = 42


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

visible_satellite_ids = []

visible_satellite_positions = []


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

        visible_satellite_ids.append(
            satellite["id"]
        )

        visible_satellite_positions.append(
            satellite_position
        )


visible_satellite_positions = np.array(
    visible_satellite_positions
)


# ------------------------------------------------------------
# 6. GENERATION DES MESURES
# ------------------------------------------------------------

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
        random_seed
)


pseudoranges = measurements[
    "pseudoranges"
]


true_clock_bias_range = (
    C_LIGHT
    * receiver_clock_bias_seconds
)


# ------------------------------------------------------------
# 7. ESTIMATION INITIALE VOLONTAIREMENT IMPARFAITE
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
# 8. SOLUTION GNSS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 9. ERREURS
# ------------------------------------------------------------

initial_position_error = np.linalg.norm(
    initial_position_guess
    - true_receiver_position
)


final_position_error = np.linalg.norm(
    estimated_position
    - true_receiver_position
)


clock_bias_range_error = (
    estimated_clock_bias_range
    - true_clock_bias_range
)


estimated_clock_bias_seconds = (
    estimated_clock_bias_range
    / C_LIGHT
)


clock_bias_time_error = (
    estimated_clock_bias_seconds
    - receiver_clock_bias_seconds
)


# ------------------------------------------------------------
# 10. PDOP A L'INSTANT DE MESURE
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


# ------------------------------------------------------------
# 11. RESULTATS
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 005-B"
)

print(
    "Positionnement GNSS par moindres carrés"
)

print(
    "=============================================="
)

print(
    f"Satellites visibles : "
    f"{len(visible_satellite_ids)}"
)

print(
    f"PDOP à cet instant : "
    f"{pdop:.3f}"
)

print()

print(
    "Convergence : "
    f"{solution['converged']}"
)

print(
    "Nombre d'itérations : "
    f"{solution['iterations']}"
)

print()

print(
    f"Erreur de position initiale : "
    f"{initial_position_error:.3f} m"
)

print(
    f"Erreur de position finale : "
    f"{final_position_error:.3f} m"
)

print()

print(
    f"Biais horloge vrai : "
    f"{true_clock_bias_range:.3f} m"
)

print(
    f"Biais horloge estimé : "
    f"{estimated_clock_bias_range:.3f} m"
)

print(
    f"Erreur sur le biais horloge : "
    f"{clock_bias_range_error:.3f} m"
)

print()

print(
    f"Biais horloge vrai : "
    f"{receiver_clock_bias_seconds * 1e6:.6f} microsecondes"
)

print(
    f"Biais horloge estimé : "
    f"{estimated_clock_bias_seconds * 1e6:.6f} microsecondes"
)

print(
    f"Erreur temporelle : "
    f"{clock_bias_time_error * 1e9:.3f} ns"
)

print()

print(
    f"RMS des résidus finaux : "
    f"{solution['final_residual_rms']:.3f} m"
)

print(
    f"Conditionnement final : "
    f"{solution['condition_number']:.3e}"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 12. HISTORIQUE DE CONVERGENCE
# ------------------------------------------------------------

iteration_numbers = [
    0
]

position_errors = [
    initial_position_error
]


for item in solution[
    "history"
]:

    iteration_numbers.append(
        item[
            "iteration"
        ]
    )

    error = np.linalg.norm(
        item[
            "position_estimate"
        ]
        - true_receiver_position
    )

    position_errors.append(
        error
    )


# ------------------------------------------------------------
# 13. FIGURE DE CONVERGENCE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.semilogy(
    iteration_numbers,
    position_errors,
    marker="o"
)

plt.xlabel(
    "Itération"
)

plt.ylabel(
    "Erreur de position [m]"
)

plt.title(
    "AURORA — Convergence du solveur GNSS"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. FIGURE DES RESIDUS FINAUX
# ------------------------------------------------------------

satellite_indices = np.arange(
    len(
        visible_satellite_ids
    )
)


plt.figure(
    figsize=(11, 5)
)

plt.scatter(
    satellite_indices,
    solution[
        "final_residuals"
    ]
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xticks(
    satellite_indices,
    visible_satellite_ids
)

plt.xlabel(
    "Satellite GPS"
)

plt.ylabel(
    "Résidu final de pseudorange [m]"
)

plt.title(
    "AURORA — Résidus après estimation GNSS"
)

plt.grid(True)

plt.tight_layout()

plt.show()