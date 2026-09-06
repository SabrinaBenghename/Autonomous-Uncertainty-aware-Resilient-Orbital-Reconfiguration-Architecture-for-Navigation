import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

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

from src.faults.gnss_faults import (
    inject_pseudorange_bias,
    select_satellite_subset
)


# ============================================================
# AURORA
# Expérience 006-A
# Impact des dégradations GNSS
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
# 2. PARAMETRES DE L'EXPERIENCE
# ------------------------------------------------------------

measurement_time = (
    120.0 * 60.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

nominal_noise_std = 3.0

degraded_noise_std = 15.0

fault_bias_meters = 100.0

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
# 6. MESURES NOMINALES
# ------------------------------------------------------------

nominal_measurements = (
    simulate_pseudorange_set(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            visible_satellite_positions,

        receiver_clock_bias_seconds=
            receiver_clock_bias_seconds,

        noise_std=
            nominal_noise_std,

        random_seed=
            random_seed
    )
)

nominal_pseudoranges = (
    nominal_measurements[
        "pseudoranges"
    ]
)


# ------------------------------------------------------------
# 7. MESURES A BRUIT FORT
# ------------------------------------------------------------

degraded_noise_measurements = (
    simulate_pseudorange_set(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            visible_satellite_positions,

        receiver_clock_bias_seconds=
            receiver_clock_bias_seconds,

        noise_std=
            degraded_noise_std,

        random_seed=
            random_seed
    )
)

degraded_noise_pseudoranges = (
    degraded_noise_measurements[
        "pseudoranges"
    ]
)


# ------------------------------------------------------------
# 8. INJECTION D'UN BIAIS DE PSEUDORANGE
# ------------------------------------------------------------

faulty_satellite_index = 0

biased_pseudoranges = (
    inject_pseudorange_bias(
        pseudoranges=
            nominal_pseudoranges,

        satellite_index=
            faulty_satellite_index,

        bias_meters=
            fault_bias_meters
    )
)

faulty_satellite_id = (
    visible_satellite_ids[
        faulty_satellite_index
    ]
)


# ------------------------------------------------------------
# 9. RECHERCHE D'UNE GEOMETRIE DEGRADEE
# ------------------------------------------------------------

# On cherche parmi toutes les combinaisons
# de 4 satellites une géométrie ayant un
# PDOP proche de cette valeur.

target_degraded_pdop = 6.0

best_subset_indices = None

best_subset_pdop = None

best_pdop_difference = np.inf


for subset in combinations(
    range(
        len(
            visible_satellite_ids
        )
    ),
    4
):

    subset_indices = np.array(
        subset,
        dtype=int
    )

    subset_positions = (
        visible_satellite_positions[
            subset_indices
        ]
    )

    dop_results = compute_dop(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            subset_positions
    )

    pdop = dop_results[
        "pdop"
    ]

    if not np.isfinite(
        pdop
    ):
        continue

    difference = abs(
        pdop
        - target_degraded_pdop
    )

    if (
        difference
        < best_pdop_difference
    ):

        best_pdop_difference = (
            difference
        )

        best_subset_indices = (
            subset_indices
        )

        best_subset_pdop = (
            pdop
        )


if best_subset_indices is None:
    raise RuntimeError(
        "Impossible de trouver un "
        "sous-ensemble GNSS valide."
    )


(
    degraded_geometry_positions,
    degraded_geometry_pseudoranges,
    degraded_geometry_ids
) = select_satellite_subset(
    satellite_positions=
        visible_satellite_positions,

    pseudoranges=
        nominal_pseudoranges,

    satellite_ids=
        visible_satellite_ids,

    selected_indices=
        best_subset_indices
)


# ------------------------------------------------------------
# 10. ESTIMATION INITIALE
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


# ------------------------------------------------------------
# 11. FONCTION D'EXECUTION D'UN SCENARIO
# ------------------------------------------------------------

def run_scenario(
    name,
    satellite_positions,
    pseudoranges,
    satellite_ids
):
    """
    Résout un scénario GNSS et retourne
    les indicateurs de performance.
    """

    dop_results = compute_dop(
        receiver_position=
            true_receiver_position,

        satellite_positions=
            satellite_positions
    )

    solution = solve_position_least_squares(
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

    estimated_position = (
        solution[
            "position"
        ]
    )

    position_error = np.linalg.norm(
        estimated_position
        - true_receiver_position
    )

    true_clock_bias_range = (
        C_LIGHT
        * receiver_clock_bias_seconds
    )

    clock_bias_error = (
        solution[
            "clock_bias_range"
        ]
        - true_clock_bias_range
    )

    return {
        "name":
            name,

        "number_of_satellites":
            len(
                satellite_ids
            ),

        "satellite_ids":
            satellite_ids,

        "pdop":
            dop_results[
                "pdop"
            ],

        "position_error":
            position_error,

        "clock_bias_error":
            clock_bias_error,

        "residual_rms":
            solution[
                "final_residual_rms"
            ],

        "iterations":
            solution[
                "iterations"
            ],

        "converged":
            solution[
                "converged"
            ],

        "condition_number":
            solution[
                "condition_number"
            ]
    }


# ------------------------------------------------------------
# 12. EXECUTION DES QUATRE SCENARIOS
# ------------------------------------------------------------

nominal_result = run_scenario(
    name=
        "Nominal",

    satellite_positions=
        visible_satellite_positions,

    pseudoranges=
        nominal_pseudoranges,

    satellite_ids=
        visible_satellite_ids
)


high_noise_result = run_scenario(
    name=
        "Bruit 15 m",

    satellite_positions=
        visible_satellite_positions,

    pseudoranges=
        degraded_noise_pseudoranges,

    satellite_ids=
        visible_satellite_ids
)


biased_result = run_scenario(
    name=
        "Biais +100 m",

    satellite_positions=
        visible_satellite_positions,

    pseudoranges=
        biased_pseudoranges,

    satellite_ids=
        visible_satellite_ids
)


degraded_geometry_result = run_scenario(
    name=
        "Seulement 4 satellites",

    satellite_positions=
        degraded_geometry_positions,

    pseudoranges=
        degraded_geometry_pseudoranges,

    satellite_ids=
        degraded_geometry_ids
)


results = [
    nominal_result,
    high_noise_result,
    biased_result,
    degraded_geometry_result
]


# ------------------------------------------------------------
# 13. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=========================================================================="
)

print(
    "AURORA — Expérience 006-A"
)

print(
    "Impact des dégradations GNSS"
)

print(
    "=========================================================================="
)

print(
    f"Satellites visibles initialement : "
    f"{len(visible_satellite_ids)}"
)

print(
    f"Satellite biaisé : GPS "
    f"{faulty_satellite_id:02d}"
)

print(
    f"Biais injecté : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"PDOP ciblé pour la géométrie dégradée : "
    f"{target_degraded_pdop:.1f}"
)

print(
    f"PDOP réellement obtenu : "
    f"{best_subset_pdop:.3f}"
)

print()

print(
    "Scénario               | Nsat | PDOP   | "
    "Erreur pos [m] | RMS résidus [m] | Conv."
)

print(
    "--------------------------------------------------------------------------"
)


for result in results:

    print(
        f"{result['name']:<22} | "
        f"{result['number_of_satellites']:>4d} | "
        f"{result['pdop']:>6.3f} | "
        f"{result['position_error']:>14.3f} | "
        f"{result['residual_rms']:>16.3f} | "
        f"{str(result['converged']):>5}"
    )


print(
    "=========================================================================="
)


# ------------------------------------------------------------
# 14. DETAILS DU SCENARIO 4 SATELLITES
# ------------------------------------------------------------

print(
    "\nSatellites conservés pour "
    "la géométrie dégradée :"
)

print(
    degraded_geometry_ids
)

print(
    f"Conditionnement : "
    f"{degraded_geometry_result['condition_number']:.3e}"
)


# ------------------------------------------------------------
# 15. FIGURE - ERREUR DE POSITION
# ------------------------------------------------------------

scenario_names = [
    result[
        "name"
    ]
    for result in results
]

position_error_values = [
    result[
        "position_error"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    scenario_names,
    position_error_values
)

plt.ylabel(
    "Erreur de position 3D [m]"
)

plt.title(
    "AURORA — Impact des dégradations "
    "sur la position GNSS"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FIGURE - RMS DES RESIDUS
# ------------------------------------------------------------

residual_values = [
    result[
        "residual_rms"
    ]
    for result in results
]


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    scenario_names,
    residual_values
)

plt.ylabel(
    "RMS des résidus [m]"
)

plt.title(
    "AURORA — Résidus GNSS selon le scénario"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()