import numpy as np


# ============================================================
# AURORA
# Phase 4 - Géométrie GNSS et DOP
# ============================================================


# ------------------------------------------------------------
# 1. VECTEUR UNITAIRE DE LIGNE DE VISEE
# ------------------------------------------------------------

def line_of_sight_unit_vector(
    receiver_position,
    satellite_position
):
    """
    Calcule le vecteur unitaire allant du
    récepteur vers le satellite GNSS.

    Parameters
    ----------
    receiver_position : ndarray
        Position du récepteur [m].

    satellite_position : ndarray
        Position du satellite GNSS [m].

    Returns
    -------
    ndarray
        Vecteur unitaire de ligne de visée.
    """

    line_of_sight = (
        satellite_position
        - receiver_position
    )

    distance = np.linalg.norm(
        line_of_sight
    )

    if distance == 0.0:
        raise ValueError(
            "Le satellite GNSS et le récepteur "
            "ne peuvent pas avoir la même position."
        )

    return (
        line_of_sight
        / distance
    )


# ------------------------------------------------------------
# 2. MATRICE DE GEOMETRIE GNSS
# ------------------------------------------------------------

def build_geometry_matrix(
    receiver_position,
    satellite_positions
):
    """
    Construit la matrice de géométrie H
    utilisée pour le calcul des DOP.

    Chaque ligne correspond à un satellite :

    [-ux, -uy, -uz, 1]

    Parameters
    ----------
    receiver_position : ndarray
        Position du récepteur [m].

    satellite_positions : ndarray
        Positions des satellites visibles,
        dimension N x 3.

    Returns
    -------
    ndarray
        Matrice H de dimension N x 4.
    """

    satellite_positions = np.asarray(
        satellite_positions
    )

    number_of_satellites = (
        satellite_positions.shape[0]
    )

    geometry_matrix = np.zeros(
        (
            number_of_satellites,
            4
        )
    )

    for index in range(
        number_of_satellites
    ):

        unit_vector = (
            line_of_sight_unit_vector(
                receiver_position,
                satellite_positions[index]
            )
        )

        geometry_matrix[
            index,
            0:3
        ] = -unit_vector

        geometry_matrix[
            index,
            3
        ] = 1.0

    return geometry_matrix


# ------------------------------------------------------------
# 3. CALCUL DES DOP
# ------------------------------------------------------------

def compute_dop(
    receiver_position,
    satellite_positions,
    condition_threshold=1e12
):
    """
    Calcule GDOP, PDOP et TDOP à partir
    de la géométrie des satellites visibles.

    Parameters
    ----------
    receiver_position : ndarray
        Position du récepteur [m].

    satellite_positions : ndarray
        Positions GNSS visibles, N x 3.

    condition_threshold : float
        Seuil au-dessus duquel la matrice
        est considérée comme trop mal
        conditionnée.

    Returns
    -------
    dict
        Résultats du calcul DOP.
    """

    satellite_positions = np.asarray(
        satellite_positions
    )

    number_of_satellites = (
        satellite_positions.shape[0]
    )

    # --------------------------------------------------------
    # Il faut au moins 4 satellites
    # --------------------------------------------------------

    if number_of_satellites < 4:

        return {
            "gdop": np.nan,
            "pdop": np.nan,
            "tdop": np.nan,
            "rank": 0,
            "condition_number": np.inf
        }

    # --------------------------------------------------------
    # Matrice de géométrie
    # --------------------------------------------------------

    geometry_matrix = (
        build_geometry_matrix(
            receiver_position,
            satellite_positions
        )
    )

    # --------------------------------------------------------
    # Rang de H
    # --------------------------------------------------------

    rank = np.linalg.matrix_rank(
        geometry_matrix
    )

    if rank < 4:

        return {
            "gdop": np.nan,
            "pdop": np.nan,
            "tdop": np.nan,
            "rank": rank,
            "condition_number": np.inf
        }

    # --------------------------------------------------------
    # Matrice normale
    # --------------------------------------------------------

    normal_matrix = (
        geometry_matrix.T
        @ geometry_matrix
    )

    condition_number = np.linalg.cond(
        normal_matrix
    )

    # --------------------------------------------------------
    # Géométrie numériquement trop mauvaise
    # --------------------------------------------------------

    if (
        not np.isfinite(
            condition_number
        )
        or condition_number
        > condition_threshold
    ):

        return {
            "gdop": np.nan,
            "pdop": np.nan,
            "tdop": np.nan,
            "rank": rank,
            "condition_number":
                condition_number
        }

    # --------------------------------------------------------
    # Matrice de covariance géométrique
    # --------------------------------------------------------

    covariance_matrix = np.linalg.inv(
        normal_matrix
    )

    # Protection contre de minuscules
    # valeurs négatives numériques
    diagonal = np.maximum(
        np.diag(
            covariance_matrix
        ),
        0.0
    )

    # --------------------------------------------------------
    # PDOP
    # --------------------------------------------------------

    pdop = np.sqrt(
        diagonal[0]
        + diagonal[1]
        + diagonal[2]
    )

    # --------------------------------------------------------
    # TDOP
    # --------------------------------------------------------

    tdop = np.sqrt(
        diagonal[3]
    )

    # --------------------------------------------------------
    # GDOP
    # --------------------------------------------------------

    gdop = np.sqrt(
        pdop**2
        + tdop**2
    )

    return {
        "gdop": gdop,
        "pdop": pdop,
        "tdop": tdop,
        "rank": rank,
        "condition_number":
            condition_number
    }