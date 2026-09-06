import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    MU_EARTH,
    REFERENCE_ORBIT_RADIUS,
    REFERENCE_ORBIT_VELOCITY,
    REFERENCE_ORBIT_PERIOD,
    initial_state,
    propagate_orbit,
)


# ============================================================
# AURORA
# Expérience 001-B
# Validation du propagateur orbital à deux corps
# ============================================================


# ------------------------------------------------------------
# 1. CALCUL DU RAYON ORBITAL
# ------------------------------------------------------------

def compute_radius(position):
    """
    Calcule la norme du vecteur position.

    Parameters
    ----------
    position : ndarray
        Tableau de taille (3, N) contenant
        x, y, z au cours du temps.

    Returns
    -------
    ndarray
        Rayon orbital r(t) [m].
    """

    return np.linalg.norm(
        position,
        axis=0
    )


# ------------------------------------------------------------
# 2. CALCUL DE LA VITESSE
# ------------------------------------------------------------

def compute_speed(velocity):
    """
    Calcule la norme du vecteur vitesse.

    Parameters
    ----------
    velocity : ndarray
        Tableau de taille (3, N) contenant
        vx, vy, vz.

    Returns
    -------
    ndarray
        Vitesse orbitale v(t) [m/s].
    """

    return np.linalg.norm(
        velocity,
        axis=0
    )


# ------------------------------------------------------------
# 3. ÉNERGIE MÉCANIQUE SPÉCIFIQUE
# ------------------------------------------------------------

def compute_specific_energy(
    radius,
    speed
):
    """
    Calcule l'énergie mécanique spécifique orbitale.

    epsilon = v^2 / 2 - mu / r

    Parameters
    ----------
    radius : ndarray
        Rayon orbital [m].

    speed : ndarray
        Vitesse orbitale [m/s].

    Returns
    -------
    ndarray
        Energie mécanique spécifique [J/kg].
    """

    kinetic_energy = (
        speed**2 / 2.0
    )

    gravitational_energy = (
        -MU_EARTH / radius
    )

    return (
        kinetic_energy
        + gravitational_energy
    )


# ------------------------------------------------------------
# 4. MOMENT CINÉTIQUE SPÉCIFIQUE
# ------------------------------------------------------------

def compute_specific_angular_momentum(
    position,
    velocity
):
    """
    Calcule le moment cinétique spécifique.

    h = r x v

    Parameters
    ----------
    position : ndarray
        Position (3, N) [m].

    velocity : ndarray
        Vitesse (3, N) [m/s].

    Returns
    -------
    ndarray
        Norme de h(t) [m^2/s].
    """

    angular_momentum_vector = np.cross(
        position.T,
        velocity.T
    )

    angular_momentum = np.linalg.norm(
        angular_momentum_vector,
        axis=1
    )

    return angular_momentum


# ------------------------------------------------------------
# 5. ERREUR RELATIVE
# ------------------------------------------------------------

def relative_error(
    values,
    reference
):
    """
    Calcule l'erreur relative par rapport
    à une valeur de référence.
    """

    return (
        (values - reference)
        / reference
    )


# ------------------------------------------------------------
# 6. AFFICHAGE DES RÉSULTATS
# ------------------------------------------------------------

def print_validation_results(
    radius,
    speed,
    energy,
    angular_momentum
):
    """
    Affiche les indicateurs numériques
    permettant de valider la simulation.
    """

    initial_radius = radius[0]
    initial_speed = speed[0]
    initial_energy = energy[0]
    initial_angular_momentum = (
        angular_momentum[0]
    )

    max_radius_error = np.max(
        np.abs(
            radius - initial_radius
        )
    )

    max_speed_error = np.max(
        np.abs(
            speed - initial_speed
        )
    )

    max_energy_relative_error = np.max(
        np.abs(
            relative_error(
                energy,
                initial_energy
            )
        )
    )

    max_angular_momentum_relative_error = (
        np.max(
            np.abs(
                relative_error(
                    angular_momentum,
                    initial_angular_momentum
                )
            )
        )
    )

    print(
        "\n"
        "=============================================="
    )

    print(
        "AURORA — Validation Expérience 001-B"
    )

    print(
        "=============================================="
    )

    print(
        f"Rayon initial : "
        f"{initial_radius / 1000:.6f} km"
    )

    print(
        f"Erreur maximale sur le rayon : "
        f"{max_radius_error:.6f} m"
    )

    print()

    print(
        f"Vitesse initiale : "
        f"{initial_speed:.6f} m/s"
    )

    print(
        f"Erreur maximale sur la vitesse : "
        f"{max_speed_error:.9f} m/s"
    )

    print()

    print(
        f"Energie spécifique initiale : "
        f"{initial_energy:.6e} J/kg"
    )

    print(
        "Erreur relative maximale "
        "sur l'énergie : "
        f"{max_energy_relative_error:.6e}"
    )

    print()

    print(
        "Moment cinétique spécifique initial : "
        f"{initial_angular_momentum:.6e} m^2/s"
    )

    print(
        "Erreur relative maximale "
        "sur le moment cinétique : "
        f"{max_angular_momentum_relative_error:.6e}"
    )

    print(
        "=============================================="
    )


# ------------------------------------------------------------
# 7. FIGURES DE VALIDATION
# ------------------------------------------------------------

def plot_radius(
    time,
    radius
):
    """
    Evolution du rayon orbital.
    """

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        time / 60.0,
        radius / 1000.0
    )

    plt.xlabel(
        "Temps [min]"
    )

    plt.ylabel(
        "Rayon orbital [km]"
    )

    plt.title(
        "AURORA — Validation du rayon orbital"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_speed(
    time,
    speed
):
    """
    Evolution de la vitesse orbitale.
    """

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        time / 60.0,
        speed / 1000.0
    )

    plt.xlabel(
        "Temps [min]"
    )

    plt.ylabel(
        "Vitesse [km/s]"
    )

    plt.title(
        "AURORA — Validation de la vitesse orbitale"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_energy_relative_error(
    time,
    energy
):
    """
    Evolution de l'erreur relative
    de l'énergie mécanique spécifique.
    """

    energy_error = relative_error(
        energy,
        energy[0]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        time / 60.0,
        energy_error
    )

    plt.xlabel(
        "Temps [min]"
    )

    plt.ylabel(
        "Erreur relative"
    )

    plt.title(
        "AURORA — Conservation de "
        "l'énergie mécanique spécifique"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_angular_momentum_relative_error(
    time,
    angular_momentum
):
    """
    Evolution de l'erreur relative
    du moment cinétique spécifique.
    """

    momentum_error = relative_error(
        angular_momentum,
        angular_momentum[0]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        time / 60.0,
        momentum_error
    )

    plt.xlabel(
        "Temps [min]"
    )

    plt.ylabel(
        "Erreur relative"
    )

    plt.title(
        "AURORA — Conservation du "
        "moment cinétique spécifique"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 8. PROGRAMME PRINCIPAL
# ------------------------------------------------------------

if __name__ == "__main__":

    # Simulation de 10 orbites pour rendre
    # la validation numérique plus exigeante.
    simulation_duration = (
        10.0
        * REFERENCE_ORBIT_PERIOD
    )

    solution = propagate_orbit(
        initial_state=initial_state,
        duration=simulation_duration,
        number_of_points=20000
    )

    time = solution.t

    # Position
    position = solution.y[0:3]

    # Vitesse
    velocity = solution.y[3:6]

    # Calcul des grandeurs physiques
    radius = compute_radius(
        position
    )

    speed = compute_speed(
        velocity
    )

    energy = compute_specific_energy(
        radius,
        speed
    )

    angular_momentum = (
        compute_specific_angular_momentum(
            position,
            velocity
        )
    )

    # Résultats numériques
    print_validation_results(
        radius,
        speed,
        energy,
        angular_momentum
    )

    # Figures
    plot_radius(
        time,
        radius
    )

    plot_speed(
        time,
        speed
    )

    plot_energy_relative_error(
        time,
        energy
    )

    plot_angular_momentum_relative_error(
        time,
        angular_momentum
    )