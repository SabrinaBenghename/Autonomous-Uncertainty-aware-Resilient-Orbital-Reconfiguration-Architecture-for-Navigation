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
    propagate_gnss_constellation,
    GPS_TOTAL_SATELLITES,
    GPS_ORBITAL_PERIOD
)

from src.navigation.gnss_visibility import (
    is_gnss_visible,
    geometric_range
)


# ============================================================
# AURORA
# Expérience 004
# Visibilité GPS depuis un CubeSat LEO
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
# 2. ETAT INITIAL DU CUBESAT
# ------------------------------------------------------------

cub_sat_position, cub_sat_velocity = (
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
        cub_sat_position,
        cub_sat_velocity
    )
)


# ------------------------------------------------------------
# 3. TEMPS DE SIMULATION
# ------------------------------------------------------------

cub_sat_period = orbital_period(
    cub_sat_semi_major_axis
)

# On simule trois orbites du CubeSat
simulation_duration = (
    3.0
    * cub_sat_period
)

number_of_points = 3000


# ------------------------------------------------------------
# 4. PROPAGATION DU CUBESAT
# ------------------------------------------------------------

cub_sat_solution = propagate_orbit(
    initial_state=cub_sat_initial_state,
    duration=simulation_duration,
    number_of_points=number_of_points
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
        duration=simulation_duration,
        number_of_points=number_of_points
    )
)


# ------------------------------------------------------------
# 6. ANALYSE DE VISIBILITE
# ------------------------------------------------------------

visible_satellite_count = np.zeros(
    number_of_points,
    dtype=int
)

minimum_visible_range = np.full(
    number_of_points,
    np.nan
)

maximum_visible_range = np.full(
    number_of_points,
    np.nan
)


for time_index in range(
    number_of_points
):

    receiver_position = (
        cub_sat_solution.y[
            0:3,
            time_index
        ]
    )

    visible_ranges = []

    for gps_solution in gps_solutions:

        gnss_position = (
            gps_solution.y[
                0:3,
                time_index
            ]
        )

        visible = is_gnss_visible(
            receiver_position,
            gnss_position
        )

        if visible:

            visible_satellite_count[
                time_index
            ] += 1

            visible_ranges.append(
                geometric_range(
                    receiver_position,
                    gnss_position
                )
            )

    if visible_ranges:

        minimum_visible_range[
            time_index
        ] = np.min(
            visible_ranges
        )

        maximum_visible_range[
            time_index
        ] = np.max(
            visible_ranges
        )


# ------------------------------------------------------------
# 7. STATISTIQUES
# ------------------------------------------------------------

minimum_visible = np.min(
    visible_satellite_count
)

maximum_visible = np.max(
    visible_satellite_count
)

mean_visible = np.mean(
    visible_satellite_count
)

availability_four_satellites = (
    np.mean(
        visible_satellite_count >= 4
    )
    * 100.0
)


# ------------------------------------------------------------
# 8. RESULTATS
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 004"
)

print(
    "Visibilité GPS depuis le CubeSat"
)

print(
    "=============================================="
)

print(
    f"Nombre total de satellites GPS simulés : "
    f"{GPS_TOTAL_SATELLITES}"
)

print(
    f"Période orbitale GPS approx. : "
    f"{GPS_ORBITAL_PERIOD / 3600:.3f} h"
)

print()

print(
    f"Minimum de satellites visibles : "
    f"{minimum_visible}"
)

print(
    f"Maximum de satellites visibles : "
    f"{maximum_visible}"
)

print(
    f"Nombre moyen visible : "
    f"{mean_visible:.2f}"
)

print()

print(
    "Disponibilité d'au moins "
    "4 satellites : "
    f"{availability_four_satellites:.2f} %"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 9. FIGURE - NOMBRE DE SATELLITES VISIBLES
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    cub_sat_solution.t / 60.0,
    visible_satellite_count
)

plt.axhline(
    4,
    linestyle="--",
    label="4 satellites"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Nombre de satellites GPS visibles"
)

plt.title(
    "AURORA — Visibilité GPS depuis le CubeSat"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 10. FIGURE - DISTANCES GNSS
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    cub_sat_solution.t / 60.0,
    minimum_visible_range / 1000.0,
    label="Distance minimale"
)

plt.plot(
    cub_sat_solution.t / 60.0,
    maximum_visible_range / 1000.0,
    label="Distance maximale"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Distance CubeSat-GPS [km]"
)

plt.title(
    "AURORA — Distances des satellites GPS visibles"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()