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


# ============================================================
# AURORA
# Expérience 005-A
# Simulation de pseudoranges GNSS
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
# 2. PARAMETRES DE MESURE GNSS
# ------------------------------------------------------------

# Instant auquel on effectue
# une mesure GNSS unique
measurement_time = (
    120.0 * 60.0
)

# Biais de l'horloge du récepteur
receiver_clock_bias_seconds = (
    100.0e-6
)

# Bruit pseudorange 1-sigma
pseudorange_noise_std = 3.0

# Permet de reproduire exactement
# les mêmes nombres aléatoires
random_seed = 42


# ------------------------------------------------------------
# 3. ETAT INITIAL DU CUBESAT
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
# 4. PROPAGATION JUSQU'A L'INSTANT DE MESURE
# ------------------------------------------------------------

cub_sat_solution = propagate_orbit(
    initial_state=cub_sat_initial_state,
    duration=measurement_time,
    number_of_points=2
)

receiver_position = (
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
        constellation=gps_constellation,
        duration=measurement_time,
        number_of_points=2
    )
)


# ------------------------------------------------------------
# 6. RECHERCHE DES SATELLITES VISIBLES
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
        receiver_position,
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
# 7. SIMULATION DES PSEUDORANGES
# ------------------------------------------------------------

measurements = simulate_pseudorange_set(
    receiver_position=
        receiver_position,

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

geometric_ranges = measurements[
    "geometric_ranges"
]

measurement_noises = measurements[
    "noises"
]

clock_bias_range = measurements[
    "clock_bias_range"
]


# ------------------------------------------------------------
# 8. RESIDUS
# ------------------------------------------------------------

raw_residuals = (
    pseudoranges
    - geometric_ranges
)

clock_corrected_residuals = (
    raw_residuals
    - clock_bias_range
)


# ------------------------------------------------------------
# 9. RESULTATS GENERAUX
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 005-A"
)

print(
    "Simulation de pseudoranges GNSS"
)

print(
    "=============================================="
)

print(
    f"Instant de mesure : "
    f"{measurement_time / 60.0:.2f} min"
)

print(
    f"Satellites GPS visibles : "
    f"{len(visible_satellite_ids)}"
)

print()

print(
    f"Biais d'horloge récepteur : "
    f"{receiver_clock_bias_seconds * 1e6:.3f} microsecondes"
)

print(
    f"Erreur de distance équivalente : "
    f"{clock_bias_range:.3f} m"
)

print(
    f"Vitesse de la lumière utilisée : "
    f"{C_LIGHT:.3f} m/s"
)

print()

print(
    f"Ecart-type du bruit imposé : "
    f"{pseudorange_noise_std:.3f} m"
)

print(
    f"Moyenne du bruit simulé : "
    f"{np.mean(measurement_noises):.3f} m"
)

print(
    f"Ecart-type du bruit simulé : "
    f"{np.std(measurement_noises):.3f} m"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 10. TABLEAU DES MESURES
# ------------------------------------------------------------

print(
    "\n"
    "ID | Distance géométrique [km] "
    "| Pseudorange [km] "
    "| Bruit [m]"
)

print(
    "------------------------------------------------------------"
)


for (
    satellite_id,
    geometric_distance,
    pseudorange,
    noise
) in zip(
    visible_satellite_ids,
    geometric_ranges,
    pseudoranges,
    measurement_noises
):

    print(
        f"{satellite_id:02d} | "
        f"{geometric_distance / 1000.0:15.3f} | "
        f"{pseudorange / 1000.0:14.3f} | "
        f"{noise:8.3f}"
    )


# ------------------------------------------------------------
# 11. VERIFICATION NUMERIQUE
# ------------------------------------------------------------

maximum_consistency_error = np.max(
    np.abs(
        clock_corrected_residuals
        - measurement_noises
    )
)

print(
    "\n"
    "Erreur maximale de cohérence du modèle : "
    f"{maximum_consistency_error:.6e} m"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 12. FIGURE :
# PSEUDORANGE - DISTANCE GEOMETRIQUE
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
    raw_residuals
)

plt.axhline(
    clock_bias_range,
    linestyle="--",
    label="Biais d'horloge théorique"
)

plt.xlabel(
    "Satellite visible"
)

plt.ylabel(
    "Pseudorange - distance géométrique [m]"
)

plt.title(
    "AURORA — Effet du biais d'horloge "
    "sur les pseudoranges"
)

plt.xticks(
    satellite_indices,
    visible_satellite_ids
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 13. FIGURE :
# BRUIT APRES CORRECTION DU BIAIS D'HORLOGE
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.scatter(
    satellite_indices,
    clock_corrected_residuals,
    label="Erreur restante"
)

plt.axhline(
    0.0,
    linestyle="--"
)

plt.xlabel(
    "Satellite visible"
)

plt.ylabel(
    "Résidu après correction horloge [m]"
)

plt.title(
    "AURORA — Bruit de pseudorange simulé"
)

plt.xticks(
    satellite_indices,
    visible_satellite_ids
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()