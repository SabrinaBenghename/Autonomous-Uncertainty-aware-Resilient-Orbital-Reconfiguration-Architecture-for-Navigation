import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# AURORA
# Phase 1 - Modèle orbital à deux corps
# CubeSat en orbite circulaire LEO à 550 km
# ============================================================


# ------------------------------------------------------------
# 1. CONSTANTES PHYSIQUES
# ------------------------------------------------------------

# Paramètre gravitationnel standard de la Terre [m^3/s^2]
MU_EARTH = 3.986004418e14

# Rayon équatorial de la Terre [m]
R_EARTH = 6_378_137.0


# ------------------------------------------------------------
# 2. PARAMÈTRES DE LA MISSION
# ------------------------------------------------------------

# Altitude de référence du CubeSat [m]
REFERENCE_ALTITUDE = 550_000.0

# Rayon orbital depuis le centre de la Terre [m]
REFERENCE_ORBIT_RADIUS = (
    R_EARTH + REFERENCE_ALTITUDE
)


# ------------------------------------------------------------
# 3. VITESSE D'UNE ORBITE CIRCULAIRE
# ------------------------------------------------------------

def circular_orbit_velocity(radius):
    """
    Calcule la vitesse nécessaire pour une orbite circulaire.

    Parameters
    ----------
    radius : float
        Distance entre le satellite et le centre de la Terre [m].

    Returns
    -------
    float
        Vitesse orbitale circulaire [m/s].
    """

    return np.sqrt(
        MU_EARTH / radius
    )


REFERENCE_ORBIT_VELOCITY = circular_orbit_velocity(
    REFERENCE_ORBIT_RADIUS
)


# ------------------------------------------------------------
# 4. PÉRIODE ORBITALE
# ------------------------------------------------------------

def orbital_period(radius):
    """
    Calcule la période d'une orbite circulaire.

    Parameters
    ----------
    radius : float
        Rayon orbital [m].

    Returns
    -------
    float
        Période orbitale [s].
    """

    return (
        2.0
        * np.pi
        * np.sqrt(
            radius**3 / MU_EARTH
        )
    )


REFERENCE_ORBIT_PERIOD = orbital_period(
    REFERENCE_ORBIT_RADIUS
)


# ------------------------------------------------------------
# 5. ÉTAT INITIAL DU CUBESAT
# ------------------------------------------------------------

# Position initiale [x, y, z] [m]
initial_position = np.array(
    [
        REFERENCE_ORBIT_RADIUS,
        0.0,
        0.0
    ]
)

# Vitesse initiale [vx, vy, vz] [m/s]
initial_velocity = np.array(
    [
        0.0,
        REFERENCE_ORBIT_VELOCITY,
        0.0
    ]
)

# Vecteur d'état complet :
# [x, y, z, vx, vy, vz]
initial_state = np.concatenate(
    (
        initial_position,
        initial_velocity
    )
)


# ------------------------------------------------------------
# 6. MODÈLE DYNAMIQUE À DEUX CORPS
# ------------------------------------------------------------

def two_body_dynamics(t, state):
    """
    Calcule la dérivée du vecteur d'état du satellite
    selon le modèle orbital à deux corps.

    Parameters
    ----------
    t : float
        Temps courant [s].

    state : ndarray
        Vecteur d'état :
        [x, y, z, vx, vy, vz]

    Returns
    -------
    ndarray
        Dérivée du vecteur d'état :
        [vx, vy, vz, ax, ay, az]
    """

    # Position du satellite
    position = state[0:3]

    # Vitesse du satellite
    velocity = state[3:6]

    # Distance satellite-centre de la Terre
    radius = np.linalg.norm(position)

    # Accélération gravitationnelle
    acceleration = (
        -MU_EARTH
        * position
        / radius**3
    )

    # Dérivée du vecteur d'état
    state_derivative = np.concatenate(
        (
            velocity,
            acceleration
        )
    )

    return state_derivative


# ------------------------------------------------------------
# 7. PROPAGATION NUMÉRIQUE DE L'ORBITE
# ------------------------------------------------------------

def propagate_orbit(
    initial_state,
    duration,
    number_of_points=5000
):
    """
    Propage numériquement l'orbite du satellite.

    Parameters
    ----------
    initial_state : ndarray
        Etat initial [x, y, z, vx, vy, vz].

    duration : float
        Durée de la simulation [s].

    number_of_points : int
        Nombre de points enregistrés.

    Returns
    -------
    OdeResult
        Solution numérique retournée par solve_ivp.
    """

    time_points = np.linspace(
        0.0,
        duration,
        number_of_points
    )

    solution = solve_ivp(
        fun=two_body_dynamics,
        t_span=(0.0, duration),
        y0=initial_state,
        t_eval=time_points,
        rtol=1e-9,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError(
            "La propagation orbitale a échoué."
        )

    return solution


# ------------------------------------------------------------
# 8. VISUALISATION
# ------------------------------------------------------------

def plot_orbit(solution):
    """
    Affiche la trajectoire orbitale dans le plan XY.

    Parameters
    ----------
    solution : OdeResult
        Solution obtenue avec propagate_orbit().
    """

    # Positions en mètres
    x = solution.y[0]
    y = solution.y[1]

    # Conversion en kilomètres pour l'affichage
    x_km = x / 1000.0
    y_km = y / 1000.0

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # Trajectoire du satellite
    ax.plot(
        x_km,
        y_km,
        label="Trajectoire du CubeSat"
    )

    # Représentation de la Terre
    earth = plt.Circle(
        (0.0, 0.0),
        R_EARTH / 1000.0,
        alpha=0.3
    )

    ax.add_patch(earth)

    # Position initiale du satellite
    ax.scatter(
        x_km[0],
        y_km[0],
        label="Position initiale"
    )

    ax.set_xlabel(
        "Position X - repère ECI [km]"
    )

    ax.set_ylabel(
        "Position Y - repère ECI [km]"
    )

    ax.set_title(
        "AURORA — Expérience 001\n"
        "CubeSat en orbite circulaire LEO à 550 km"
    )

    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()


# ------------------------------------------------------------
# 9. PROGRAMME PRINCIPAL
# ------------------------------------------------------------

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "AURORA — Expérience 001"
    )

    print(
        "Simulation orbitale d'un CubeSat"
    )

    print(
        "=========================================="
    )

    print(
        f"Altitude : "
        f"{REFERENCE_ALTITUDE / 1000:.3f} km"
    )

    print(
        f"Rayon orbital : "
        f"{REFERENCE_ORBIT_RADIUS / 1000:.3f} km"
    )

    print(
        f"Vitesse circulaire : "
        f"{REFERENCE_ORBIT_VELOCITY / 1000:.3f} km/s"
    )

    print(
        f"Période orbitale : "
        f"{REFERENCE_ORBIT_PERIOD / 60:.2f} minutes"
    )

    print(
        "=========================================="
    )

    # On simule deux orbites complètes
    simulation_duration = (
        2.0 * REFERENCE_ORBIT_PERIOD
    )

    solution = propagate_orbit(
        initial_state=initial_state,
        duration=simulation_duration
    )

    print(
        f"Simulation réussie : "
        f"{solution.success}"
    )

    # Affichage de l'orbite
    plot_orbit(solution)