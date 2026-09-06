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
    GPS_TOTAL_SATELLITES
)

from src.navigation.gnss_visibility import (
    is_gnss_visible
)

from src.navigation.gnss_geometry import (
    compute_dop
)


# ============================================================
# AURORA
# Expérience 004-B
# Géométrie GNSS et PDOP
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
# 6. TABLEAUX DE RESULTATS
# ------------------------------------------------------------

visible_satellite_count = np.zeros(
    number_of_points,
    dtype=int
)

pdop_values = np.full(
    number_of_points,
    np.nan
)

gdop_values = np.full(
    number_of_points,
    np.nan
)

tdop_values = np.full(
    number_of_points,
    np.nan
)

condition_numbers = np.full(
    number_of_points,
    np.nan
)


# ------------------------------------------------------------
# 7. ANALYSE DE LA GEOMETRIE
# ------------------------------------------------------------

for time_index in range(
    number_of_points
):

    receiver_position = (
        cub_sat_solution.y[
            0:3,
            time_index
        ]
    )

    visible_positions = []

    for gps_solution in gps_solutions:

        satellite_position = (
            gps_solution.y[
                0:3,
                time_index
            ]
        )

        if is_gnss_visible(
            receiver_position,
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

    if len(
        visible_positions
    ) >= 4:

        visible_positions = np.array(
            visible_positions
        )

        dop_results = compute_dop(
            receiver_position,
            visible_positions
        )

        pdop_values[
            time_index
        ] = dop_results[
            "pdop"
        ]

        gdop_values[
            time_index
        ] = dop_results[
            "gdop"
        ]

        tdop_values[
            time_index
        ] = dop_results[
            "tdop"
        ]

        condition_numbers[
            time_index
        ] = dop_results[
            "condition_number"
        ]


# ------------------------------------------------------------
# 8. ECHANTILLONS VALIDES
# ------------------------------------------------------------

valid_pdop_mask = np.isfinite(
    pdop_values
)

valid_pdop = pdop_values[
    valid_pdop_mask
]

valid_gdop = gdop_values[
    np.isfinite(
        gdop_values
    )
]

valid_tdop = tdop_values[
    np.isfinite(
        tdop_values
    )
]


# ------------------------------------------------------------
# 9. STATISTIQUES
# ------------------------------------------------------------

if len(valid_pdop) == 0:

    raise RuntimeError(
        "Aucune géométrie GNSS valide "
        "n'a été trouvée."
    )


minimum_pdop = np.min(
    valid_pdop
)

maximum_pdop = np.max(
    valid_pdop
)

mean_pdop = np.mean(
    valid_pdop
)

median_pdop = np.median(
    valid_pdop
)


minimum_gdop = np.min(
    valid_gdop
)

maximum_gdop = np.max(
    valid_gdop
)

mean_gdop = np.mean(
    valid_gdop
)


mean_tdop = np.mean(
    valid_tdop
)


pdop_below_2 = (
    np.mean(
        valid_pdop < 2.0
    )
    * 100.0
)

pdop_below_3 = (
    np.mean(
        valid_pdop < 3.0
    )
    * 100.0
)

valid_geometry_percentage = (
    np.mean(
        valid_pdop_mask
    )
    * 100.0
)


# ------------------------------------------------------------
# 10. INSTANT DU PIRE PDOP
# ------------------------------------------------------------

worst_pdop_index = np.nanargmax(
    pdop_values
)

worst_pdop_time = (
    cub_sat_solution.t[
        worst_pdop_index
    ]
    / 60.0
)

worst_visible_count = (
    visible_satellite_count[
        worst_pdop_index
    ]
)


# ------------------------------------------------------------
# 11. RESULTATS NUMERIQUES
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 004-B"
)

print(
    "Géométrie GNSS et Dilution of Precision"
)

print(
    "=============================================="
)

print(
    f"Satellites GPS simulés : "
    f"{GPS_TOTAL_SATELLITES}"
)

print()

print(
    f"PDOP minimum : "
    f"{minimum_pdop:.3f}"
)

print(
    f"PDOP moyen : "
    f"{mean_pdop:.3f}"
)

print(
    f"PDOP médian : "
    f"{median_pdop:.3f}"
)

print(
    f"PDOP maximum : "
    f"{maximum_pdop:.3f}"
)

print()

print(
    f"GDOP minimum : "
    f"{minimum_gdop:.3f}"
)

print(
    f"GDOP moyen : "
    f"{mean_gdop:.3f}"
)

print(
    f"GDOP maximum : "
    f"{maximum_gdop:.3f}"
)

print()

print(
    f"TDOP moyen : "
    f"{mean_tdop:.3f}"
)

print()

print(
    f"Pourcentage PDOP < 2 : "
    f"{pdop_below_2:.2f} %"
)

print(
    f"Pourcentage PDOP < 3 : "
    f"{pdop_below_3:.2f} %"
)

print(
    f"Géométrie calculable : "
    f"{valid_geometry_percentage:.2f} %"
)

print()

print(
    f"Pire PDOP à t = "
    f"{worst_pdop_time:.2f} min"
)

print(
    f"Satellites visibles à cet instant : "
    f"{worst_visible_count}"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 12. FIGURE PDOP
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    cub_sat_solution.t / 60.0,
    pdop_values,
    label="PDOP"
)

plt.axhline(
    2.0,
    linestyle="--",
    label="PDOP = 2"
)

plt.axhline(
    3.0,
    linestyle=":",
    label="PDOP = 3"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "PDOP [-]"
)

plt.title(
    "AURORA — Qualité de la géométrie GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 13. FIGURE PDOP VS SATELLITES VISIBLES
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    visible_satellite_count[
        valid_pdop_mask
    ],
    valid_pdop,
    s=12,
    alpha=0.6
)

plt.xlabel(
    "Nombre de satellites GPS visibles"
)

plt.ylabel(
    "PDOP [-]"
)

plt.title(
    "AURORA — PDOP vs nombre de satellites visibles"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. FIGURE GDOP ET TDOP
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    cub_sat_solution.t / 60.0,
    gdop_values,
    label="GDOP"
)

plt.plot(
    cub_sat_solution.t / 60.0,
    tdop_values,
    label="TDOP"
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "DOP [-]"
)

plt.title(
    "AURORA — GDOP et TDOP"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()