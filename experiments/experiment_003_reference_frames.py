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
    OMEGA_EARTH,
    eci_to_ecef_position,
    ecef_to_eci_position,
    eci_to_ecef_velocity,
    ecef_to_eci_velocity
)


# ============================================================
# AURORA
# Expérience 003
# Transformation ECI <-> ECEF
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE DE REFERENCE
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
# 3. PROPAGATION
# ------------------------------------------------------------

period = orbital_period(
    semi_major_axis
)

simulation_duration = (
    3.0 * period
)

solution = propagate_orbit(
    initial_state=initial_state,
    duration=simulation_duration,
    number_of_points=6000
)


# ------------------------------------------------------------
# 4. STOCKAGE DES COORDONNEES
# ------------------------------------------------------------

number_of_points = len(
    solution.t
)

positions_ecef = np.zeros(
    (
        3,
        number_of_points
    )
)

velocities_ecef = np.zeros(
    (
        3,
        number_of_points
    )
)


# ------------------------------------------------------------
# 5. TRANSFORMATION ECI -> ECEF
# ------------------------------------------------------------

for index, time in enumerate(
    solution.t
):

    position_eci = (
        solution.y[0:3, index]
    )

    velocity_eci = (
        solution.y[3:6, index]
    )

    positions_ecef[:, index] = (
        eci_to_ecef_position(
            position_eci,
            time
        )
    )

    velocities_ecef[:, index] = (
        eci_to_ecef_velocity(
            position_eci,
            velocity_eci,
            time
        )
    )


# ------------------------------------------------------------
# 6. TEST DE REVERSIBILITE
# ------------------------------------------------------------

max_position_roundtrip_error = 0.0

max_velocity_roundtrip_error = 0.0


for index, time in enumerate(
    solution.t
):

    position_eci_original = (
        solution.y[0:3, index]
    )

    velocity_eci_original = (
        solution.y[3:6, index]
    )

    position_ecef = (
        positions_ecef[:, index]
    )

    velocity_ecef = (
        velocities_ecef[:, index]
    )

    recovered_position_eci = (
        ecef_to_eci_position(
            position_ecef,
            time
        )
    )

    recovered_velocity_eci = (
        ecef_to_eci_velocity(
            position_ecef,
            velocity_ecef,
            time
        )
    )

    position_error = np.linalg.norm(
        recovered_position_eci
        - position_eci_original
    )

    velocity_error = np.linalg.norm(
        recovered_velocity_eci
        - velocity_eci_original
    )

    max_position_roundtrip_error = max(
        max_position_roundtrip_error,
        position_error
    )

    max_velocity_roundtrip_error = max(
        max_velocity_roundtrip_error,
        velocity_error
    )


# ------------------------------------------------------------
# 7. VERIFICATION DE LA NORME
# ------------------------------------------------------------

radius_eci = np.linalg.norm(
    solution.y[0:3],
    axis=0
)

radius_ecef = np.linalg.norm(
    positions_ecef,
    axis=0
)

max_radius_difference = np.max(
    np.abs(
        radius_eci
        - radius_ecef
    )
)


# ------------------------------------------------------------
# 8. RESULTATS NUMERIQUES
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 003"
)

print(
    "Transformation ECI <-> ECEF"
)

print(
    "=============================================="
)

print(
    f"Vitesse angulaire terrestre : "
    f"{OMEGA_EARTH:.10e} rad/s"
)

print()

print(
    "Erreur maximale aller-retour "
    "sur la position : "
    f"{max_position_roundtrip_error:.6e} m"
)

print(
    "Erreur maximale aller-retour "
    "sur la vitesse : "
    f"{max_velocity_roundtrip_error:.6e} m/s"
)

print()

print(
    "Différence maximale entre "
    "||r_ECI|| et ||r_ECEF|| : "
    f"{max_radius_difference:.6e} m"
)

print(
    "=============================================="
)


# ------------------------------------------------------------
# 9. FIGURE ECI
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    solution.y[0] / 1000.0,
    solution.y[1] / 1000.0,
    solution.y[2] / 1000.0
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
    "AURORA — Trajectoire dans le repère ECI"
)

ax.set_box_aspect(
    [1, 1, 1]
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 10. FIGURE ECEF
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    positions_ecef[0] / 1000.0,
    positions_ecef[1] / 1000.0,
    positions_ecef[2] / 1000.0
)

ax.set_xlabel(
    "X ECEF [km]"
)

ax.set_ylabel(
    "Y ECEF [km]"
)

ax.set_zlabel(
    "Z ECEF [km]"
)

ax.set_title(
    "AURORA — Trajectoire vue depuis "
    "le repère terrestre ECEF"
)

ax.set_box_aspect(
    [1, 1, 1]
)

plt.tight_layout()

plt.show()