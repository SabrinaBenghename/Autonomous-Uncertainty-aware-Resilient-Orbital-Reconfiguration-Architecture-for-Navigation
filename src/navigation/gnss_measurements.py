import numpy as np

from src.navigation.gnss_visibility import (
    geometric_range
)


# ============================================================
# AURORA
# Phase 5 - Mesures GNSS
# Modèle de pseudorange
# ============================================================


# ------------------------------------------------------------
# 1. VITESSE DE LA LUMIERE
# ------------------------------------------------------------

C_LIGHT = 299_792_458.0  # [m/s]


# ------------------------------------------------------------
# 2. BIAIS D'HORLOGE EN METRES
# ------------------------------------------------------------

def receiver_clock_bias_to_range(
    clock_bias_seconds
):
    """
    Convertit un biais d'horloge [s]
    en erreur équivalente de distance [m].
    """

    return (
        C_LIGHT
        * clock_bias_seconds
    )


# ------------------------------------------------------------
# 3. SIMULATION D'UNE PSEUDORANGE
# ------------------------------------------------------------

def simulate_pseudorange(
    receiver_position,
    satellite_position,
    receiver_clock_bias_seconds=0.0,
    noise_std=0.0,
    rng=None
):
    """
    Simule une mesure de pseudorange.

    Modèle :
        rho = geometric_range
              + c * delta_t
              + noise

    Parameters
    ----------
    receiver_position : ndarray
        Position vraie du récepteur [m].

    satellite_position : ndarray
        Position du satellite GNSS [m].

    receiver_clock_bias_seconds : float
        Biais d'horloge du récepteur [s].

    noise_std : float
        Ecart-type du bruit de mesure [m].

    rng : numpy random generator
        Générateur aléatoire.

    Returns
    -------
    dict
        Pseudorange et composantes associées.
    """

    if noise_std < 0.0:
        raise ValueError(
            "noise_std doit être positif ou nul."
        )

    if rng is None:
        rng = np.random.default_rng()

    true_geometric_range = geometric_range(
        receiver_position,
        satellite_position
    )

    clock_bias_range = (
        receiver_clock_bias_to_range(
            receiver_clock_bias_seconds
        )
    )

    measurement_noise = rng.normal(
        loc=0.0,
        scale=noise_std
    )

    pseudorange = (
        true_geometric_range
        + clock_bias_range
        + measurement_noise
    )

    return {
        "pseudorange":
            pseudorange,

        "geometric_range":
            true_geometric_range,

        "clock_bias_range":
            clock_bias_range,

        "noise":
            measurement_noise
    }


# ------------------------------------------------------------
# 4. SIMULATION D'UN ENSEMBLE DE PSEUDORANGES
# ------------------------------------------------------------

def simulate_pseudorange_set(
    receiver_position,
    satellite_positions,
    receiver_clock_bias_seconds=0.0,
    noise_std=0.0,
    random_seed=None
):
    """
    Simule les pseudoranges vers plusieurs
    satellites GNSS visibles.

    Parameters
    ----------
    receiver_position : ndarray
        Position vraie du récepteur [m].

    satellite_positions : ndarray
        Tableau N x 3 des positions GNSS [m].

    receiver_clock_bias_seconds : float
        Biais d'horloge récepteur [s].

    noise_std : float
        Ecart-type du bruit [m].

    random_seed : int or None
        Graine permettant de reproduire
        exactement l'expérience.

    Returns
    -------
    dict
        Tableau des pseudoranges et
        composantes d'erreur.
    """

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    if (
        satellite_positions.ndim != 2
        or satellite_positions.shape[1] != 3
    ):
        raise ValueError(
            "satellite_positions doit avoir "
            "la dimension N x 3."
        )

    rng = np.random.default_rng(
        random_seed
    )

    number_of_satellites = (
        satellite_positions.shape[0]
    )

    pseudoranges = np.zeros(
        number_of_satellites
    )

    geometric_ranges = np.zeros(
        number_of_satellites
    )

    noises = np.zeros(
        number_of_satellites
    )

    clock_bias_range = (
        receiver_clock_bias_to_range(
            receiver_clock_bias_seconds
        )
    )

    for index in range(
        number_of_satellites
    ):

        measurement = simulate_pseudorange(
            receiver_position=
                receiver_position,

            satellite_position=
                satellite_positions[index],

            receiver_clock_bias_seconds=
                receiver_clock_bias_seconds,

            noise_std=
                noise_std,

            rng=
                rng
        )

        pseudoranges[index] = (
            measurement[
                "pseudorange"
            ]
        )

        geometric_ranges[index] = (
            measurement[
                "geometric_range"
            ]
        )

        noises[index] = (
            measurement[
                "noise"
            ]
        )

    return {
        "pseudoranges":
            pseudoranges,

        "geometric_ranges":
            geometric_ranges,

        "clock_bias_range":
            clock_bias_range,

        "noises":
            noises
    }