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

from src.navigation.reference_frames import (
    eci_to_ecef_position
)

from src.navigation.geodesy import (
    ecef_to_geodetic,
    geodetic_to_ecef
)


# ============================================================
# AURORA
# Expérience 003-B
# Ground Track du CubeSat
# ============================================================


# ------------------------------------------------------------
# 1. ELEMENTS ORBITAUX
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
# 2. ETAT INITIAL
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
# 3. PROPAGATION ORBITALE
# ------------------------------------------------------------

period = orbital_period(
    semi_major_axis
)

# Cinq orbites pour bien observer
# le déplacement de la trace au sol.
simulation_duration = (
    5.0 * period
)

solution = propagate_orbit(
    initial_state=initial_state,
    duration=simulation_duration,
    number_of_points=10000
)


# ------------------------------------------------------------
# 4. TABLEAUX DE RESULTATS
# ------------------------------------------------------------

number_of_points = len(
    solution.t
)

latitudes = np.zeros(
    number_of_points
)

longitudes = np.zeros(
    number_of_points
)

altitudes = np.zeros(
    number_of_points
)


max_roundtrip_error = 0.0


# ------------------------------------------------------------
# 5. ECI -> ECEF -> GEODETIQUE
# ------------------------------------------------------------

for index, time in enumerate(
    solution.t
):

    position_eci = (
        solution.y[0:3, index]
    )

    # Position dans le repère terrestre
    position_ecef = (
        eci_to_ecef_position(
            position_eci,
            time
        )
    )

    # Conversion vers
    # latitude / longitude / altitude
    (
        latitude,
        longitude,
        altitude
    ) = ecef_to_geodetic(
        position_ecef
    )

    latitudes[index] = latitude

    longitudes[index] = longitude

    altitudes[index] = altitude

    # --------------------------------------------------------
    # Validation :
    # ECEF -> géodésique -> ECEF
    # --------------------------------------------------------

    recovered_position_ecef = (
        geodetic_to_ecef(
            latitude,
            longitude,
            altitude
        )
    )

    roundtrip_error = np.linalg.norm(
        recovered_position_ecef
        - position_ecef
    )

    max_roundtrip_error = max(
        max_roundtrip_error,
        roundtrip_error
    )


# ------------------------------------------------------------
# 6. CONVERSION EN DEGRES / KM
# ------------------------------------------------------------

latitudes_deg = np.rad2deg(
    latitudes
)

longitudes_deg = np.rad2deg(
    longitudes
)

altitudes_km = (
    altitudes / 1000.0
)


# ------------------------------------------------------------
# 7. RESULTATS NUMERIQUES
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 003-B"
)

print(
    "Ground Track et coordonnées géodésiques"
)

print(
    "=============================================="
)

print(
    f"Latitude minimale : "
    f"{np.min(latitudes_deg):.3f} deg"
)

print(
    f"Latitude maximale : "
    f"{np.max(latitudes_deg):.3f} deg"
)

print()

print(
    f"Altitude minimale : "
    f"{np.min(altitudes_km):.3f} km"
)

print(
    f"Altitude maximale : "
    f"{np.max(altitudes_km):.3f} km"
)

print()

print(
    "Erreur maximale ECEF -> géodésique "
    "-> ECEF : "
    f"{max_roundtrip_error:.6e} m"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 8. DETECTION DES SAUTS DE LONGITUDE
# ------------------------------------------------------------

longitude_jumps = np.abs(
    np.diff(
        longitudes_deg
    )
)

break_indices = np.where(
    longitude_jumps > 180.0
)[0]


plot_longitudes = (
    longitudes_deg.copy()
)

plot_latitudes = (
    latitudes_deg.copy()
)


# On insère des NaN après un passage
# de +180° à -180° afin d'éviter
# une ligne artificielle sur la figure.

for index in break_indices:

    plot_longitudes[
        index + 1
    ] = np.nan

    plot_latitudes[
        index + 1
    ] = np.nan


# ------------------------------------------------------------
# 9. FIGURE : GROUND TRACK
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    plot_longitudes,
    plot_latitudes,
    linewidth=1.0
)

plt.scatter(
    longitudes_deg[0],
    latitudes_deg[0],
    label="Position initiale"
)

plt.xlabel(
    "Longitude [deg]"
)

plt.ylabel(
    "Latitude [deg]"
)

plt.title(
    "AURORA — Expérience 003-B\n"
    "Trace au sol du CubeSat"
)

plt.xlim(
    -180,
    180
)

plt.ylim(
    -90,
    90
)

plt.xticks(
    np.arange(
        -180,
        181,
        30
    )
)

plt.yticks(
    np.arange(
        -90,
        91,
        30
    )
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 10. FIGURE : ALTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    solution.t / 60.0,
    altitudes_km
)

plt.xlabel(
    "Temps [min]"
)

plt.ylabel(
    "Altitude géodésique [km]"
)

plt.title(
    "AURORA — Evolution de l'altitude du CubeSat"
)

plt.grid(True)

plt.tight_layout()

plt.show()