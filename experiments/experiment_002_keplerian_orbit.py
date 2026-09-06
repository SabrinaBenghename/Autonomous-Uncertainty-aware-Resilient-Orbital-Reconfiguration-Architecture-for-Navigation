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


# ============================================================
# AURORA
# Expérience 002
# Orbite 3D définie par éléments képlériens
# ============================================================


# ------------------------------------------------------------
# 1. ELEMENTS ORBITAUX
# ------------------------------------------------------------

# Demi-grand axe
semi_major_axis = (
    R_EARTH + 550_000.0
)

# Faible excentricité
eccentricity = 0.01

# Orbite fortement inclinée / quasi polaire
inclination = np.deg2rad(
    97.6
)

# Ascension droite du noeud ascendant
raan = np.deg2rad(
    40.0
)

# Argument du périgée
argument_of_periapsis = np.deg2rad(
    30.0
)

# Position initiale sur l'orbite
true_anomaly = np.deg2rad(
    0.0
)


# ------------------------------------------------------------
# 2. CONVERSION KEPLER -> CARTESIEN
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
# 3. PERIODE ORBITALE
# ------------------------------------------------------------

period = orbital_period(
    semi_major_axis
)


# ------------------------------------------------------------
# 4. PROPAGATION
# ------------------------------------------------------------

solution = propagate_orbit(
    initial_state=initial_state,
    duration=period,
    number_of_points=5000
)


# ------------------------------------------------------------
# 5. RESULTATS
# ------------------------------------------------------------

print(
    "=========================================="
)

print(
    "AURORA — Expérience 002"
)

print(
    "Orbite définie par éléments képlériens"
)

print(
    "=========================================="
)

print(
    f"Demi-grand axe : "
    f"{semi_major_axis / 1000:.3f} km"
)

print(
    f"Excentricité : "
    f"{eccentricity:.4f}"
)

print(
    f"Inclinaison : "
    f"{np.rad2deg(inclination):.3f} deg"
)

print(
    f"RAAN : "
    f"{np.rad2deg(raan):.3f} deg"
)

print(
    f"Argument du périgée : "
    f"{np.rad2deg(argument_of_periapsis):.3f} deg"
)

print(
    f"Période : "
    f"{period / 60:.2f} min"
)

print(
    "=========================================="
)


# ------------------------------------------------------------
# 6. TRAJECTOIRE
# ------------------------------------------------------------

x = solution.y[0] / 1000.0
y = solution.y[1] / 1000.0
z = solution.y[2] / 1000.0


# ------------------------------------------------------------
# 7. VISUALISATION 3D
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    x,
    y,
    z,
    label="Orbite du CubeSat"
)

ax.scatter(
    x[0],
    y[0],
    z[0],
    label="Position initiale"
)


# Représentation de la Terre

u = np.linspace(
    0,
    2 * np.pi,
    50
)

v = np.linspace(
    0,
    np.pi,
    25
)

earth_x = (
    R_EARTH
    / 1000.0
    * np.outer(
        np.cos(u),
        np.sin(v)
    )
)

earth_y = (
    R_EARTH
    / 1000.0
    * np.outer(
        np.sin(u),
        np.sin(v)
    )
)

earth_z = (
    R_EARTH
    / 1000.0
    * np.outer(
        np.ones_like(u),
        np.cos(v)
    )
)

ax.plot_wireframe(
    earth_x,
    earth_y,
    earth_z,
    linewidth=0.3,
    alpha=0.25
)


ax.set_xlabel(
    "X ECI [km]"
)

ax.set_ylabel(
    "Y ECI [km]"
)

ax.set_zlabel(
    "Z ECI [km]"
)

ax.set_title(
    "AURORA — Expérience 002\n"
    "Orbite LEO 3D définie par éléments képlériens"
)

ax.legend()
ax.set_box_aspect(
    [1, 1, 1]
)
plt.tight_layout()

plt.show()