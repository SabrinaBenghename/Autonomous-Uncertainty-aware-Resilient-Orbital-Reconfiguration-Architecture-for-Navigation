import numpy as np

from src.dynamics.orbit import R_EARTH


# ============================================================
# AURORA
# Phase 4 - Visibilité GNSS
# ============================================================


def minimum_distance_to_earth(
    receiver_position,
    gnss_position
):
    """
    Calcule la distance minimale entre
    le segment récepteur-GNSS et le centre
    de la Terre.

    Parameters
    ----------
    receiver_position : ndarray
        Position du CubeSat [m].

    gnss_position : ndarray
        Position du satellite GNSS [m].

    Returns
    -------
    float
        Distance minimale [m].

    float
        Paramètre du point le plus proche
        sur le segment.
    """

    line_vector = (
        gnss_position
        - receiver_position
    )

    denominator = np.dot(
        line_vector,
        line_vector
    )

    if denominator == 0.0:
        return (
            np.linalg.norm(
                receiver_position
            ),
            0.0
        )

    parameter = (
        -np.dot(
            receiver_position,
            line_vector
        )
        / denominator
    )

    # Le point doit rester sur le segment
    parameter_clamped = np.clip(
        parameter,
        0.0,
        1.0
    )

    closest_point = (
        receiver_position
        +
        parameter_clamped
        * line_vector
    )

    minimum_distance = np.linalg.norm(
        closest_point
    )

    return (
        minimum_distance,
        parameter_clamped
    )


def is_gnss_visible(
    receiver_position,
    gnss_position
):
    """
    Détermine si la Terre bloque ou non
    la ligne de visée entre le CubeSat
    et un satellite GNSS.

    Returns
    -------
    bool
        True si le GNSS est visible.
        False si la Terre occulte le signal.
    """

    minimum_distance, parameter = (
        minimum_distance_to_earth(
            receiver_position,
            gnss_position
        )
    )

    earth_blocks_signal = (
        0.0 < parameter < 1.0
        and minimum_distance
        < R_EARTH
    )

    return not earth_blocks_signal


def geometric_range(
    receiver_position,
    gnss_position
):
    """
    Distance géométrique entre
    le récepteur et le satellite GNSS.
    """

    return np.linalg.norm(
        gnss_position
        - receiver_position
    )