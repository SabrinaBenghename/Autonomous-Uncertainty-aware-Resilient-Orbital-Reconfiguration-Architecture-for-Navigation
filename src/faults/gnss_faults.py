import numpy as np


# ============================================================
# AURORA
# Phase 6 - Défauts et dégradations GNSS
# ============================================================


# ------------------------------------------------------------
# 1. BIAIS SUR UNE PSEUDORANGE
# ------------------------------------------------------------

def inject_pseudorange_bias(
    pseudoranges,
    satellite_index,
    bias_meters
):
    """
    Injecte un biais constant sur une mesure
    de pseudorange GNSS.

    Parameters
    ----------
    pseudoranges : ndarray
        Pseudoranges originales [m].

    satellite_index : int
        Index du satellite fauté.

    bias_meters : float
        Biais ajouté [m].

    Returns
    -------
    ndarray
        Nouvelles pseudoranges fautées.
    """

    faulty_pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    ).copy()

    if (
        satellite_index < 0
        or satellite_index >= len(
            faulty_pseudoranges
        )
    ):
        raise IndexError(
            "satellite_index invalide."
        )

    faulty_pseudoranges[
        satellite_index
    ] += bias_meters

    return faulty_pseudoranges


# ------------------------------------------------------------
# 2. EXTRACTION D'UN SOUS-ENSEMBLE DE SATELLITES
# ------------------------------------------------------------

def select_satellite_subset(
    satellite_positions,
    pseudoranges,
    satellite_ids,
    selected_indices
):
    """
    Conserve uniquement un sous-ensemble
    de satellites GNSS.

    Parameters
    ----------
    satellite_positions : ndarray
        Positions GNSS N x 3.

    pseudoranges : ndarray
        Mesures GNSS [m].

    satellite_ids : list
        Identifiants satellites.

    selected_indices : iterable
        Indices des satellites conservés.

    Returns
    -------
    tuple
        positions, pseudoranges, ids
        du sous-ensemble sélectionné.
    """

    selected_indices = np.asarray(
        selected_indices,
        dtype=int
    )

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )

    selected_positions = (
        satellite_positions[
            selected_indices
        ]
    )

    selected_pseudoranges = (
        pseudoranges[
            selected_indices
        ]
    )

    selected_ids = [
        satellite_ids[index]
        for index in selected_indices
    ]

    return (
        selected_positions,
        selected_pseudoranges,
        selected_ids
    )