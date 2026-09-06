import numpy as np


# ============================================================
# AURORA
# Phase 5 - Positionnement GNSS
# Moindres carrés itératifs
# ============================================================


# ------------------------------------------------------------
# 1. PREDICTION DES PSEUDORANGES
# ------------------------------------------------------------

def predict_pseudoranges(
    receiver_position,
    satellite_positions,
    clock_bias_range
):
    """
    Prédit les pseudoranges à partir d'une
    estimation de position et de biais d'horloge.

    Parameters
    ----------
    receiver_position : ndarray
        Position estimée du récepteur [m].

    satellite_positions : ndarray
        Positions GNSS, dimension N x 3 [m].

    clock_bias_range : float
        Biais d'horloge exprimé en mètres :
        b = c * delta_t.

    Returns
    -------
    predicted_pseudoranges : ndarray
        Pseudoranges prédites [m].

    geometric_ranges : ndarray
        Distances géométriques [m].
    """

    receiver_position = np.asarray(
        receiver_position,
        dtype=float
    )

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    relative_vectors = (
        satellite_positions
        - receiver_position
    )

    geometric_ranges = np.linalg.norm(
        relative_vectors,
        axis=1
    )

    if np.any(
        geometric_ranges <= 0.0
    ):
        raise ValueError(
            "Distance GNSS-récepteur invalide."
        )

    predicted_pseudoranges = (
        geometric_ranges
        + clock_bias_range
    )

    return (
        predicted_pseudoranges,
        geometric_ranges
    )


# ------------------------------------------------------------
# 2. MATRICE JACOBIENNE
# ------------------------------------------------------------

def build_pseudorange_jacobian(
    receiver_position,
    satellite_positions
):
    """
    Construit la matrice H du problème GNSS.

    Chaque ligne est :

    [-ux, -uy, -uz, 1]

    où u est le vecteur unitaire allant
    du récepteur vers le satellite.

    Returns
    -------
    ndarray
        Matrice H, dimension N x 4.
    """

    receiver_position = np.asarray(
        receiver_position,
        dtype=float
    )

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    relative_vectors = (
        satellite_positions
        - receiver_position
    )

    geometric_ranges = np.linalg.norm(
        relative_vectors,
        axis=1
    )

    if np.any(
        geometric_ranges <= 0.0
    ):
        raise ValueError(
            "Distance GNSS-récepteur invalide."
        )

    line_of_sight_vectors = (
        relative_vectors
        / geometric_ranges[:, None]
    )

    number_of_satellites = (
        satellite_positions.shape[0]
    )

    H = np.zeros(
        (
            number_of_satellites,
            4
        )
    )

    H[:, 0:3] = (
        -line_of_sight_vectors
    )

    H[:, 3] = 1.0

    return H


# ------------------------------------------------------------
# 3. SOLVEUR ITERATIF
# ------------------------------------------------------------

def solve_position_least_squares(
    satellite_positions,
    pseudoranges,
    initial_position,
    initial_clock_bias_range=0.0,
    tolerance=1e-4,
    max_iterations=20
):
    """
    Estime la position du récepteur GNSS
    et son biais d'horloge par moindres
    carrés itératifs.

    Etat estimé :

        x = [X, Y, Z, b]

    avec :

        b = c * delta_t

    Parameters
    ----------
    satellite_positions : ndarray
        Positions GNSS visibles N x 3 [m].

    pseudoranges : ndarray
        Mesures de pseudorange [m].

    initial_position : ndarray
        Première estimation de position [m].

    initial_clock_bias_range : float
        Première estimation de b [m].

    tolerance : float
        Seuil de convergence [m].

    max_iterations : int
        Nombre maximal d'itérations.

    Returns
    -------
    dict
        Solution GNSS et informations
        de convergence.
    """

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )

    position_estimate = np.asarray(
        initial_position,
        dtype=float
    ).copy()

    clock_bias_range_estimate = float(
        initial_clock_bias_range
    )

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if (
        satellite_positions.ndim != 2
        or satellite_positions.shape[1] != 3
    ):
        raise ValueError(
            "satellite_positions doit être N x 3."
        )

    if (
        satellite_positions.shape[0]
        != pseudoranges.shape[0]
    ):
        raise ValueError(
            "Le nombre de satellites et "
            "de pseudoranges doit être identique."
        )

    if (
        satellite_positions.shape[0]
        < 4
    ):
        raise ValueError(
            "Au moins 4 satellites sont nécessaires."
        )

    history = []

    converged = False

    # --------------------------------------------------------
    # Boucle de Gauss-Newton
    # --------------------------------------------------------

    for iteration in range(
        1,
        max_iterations + 1
    ):

        # ----------------------------------------------------
        # Pseudoranges prédites
        # ----------------------------------------------------

        (
            predicted_pseudoranges,
            _
        ) = predict_pseudoranges(
            receiver_position=
                position_estimate,

            satellite_positions=
                satellite_positions,

            clock_bias_range=
                clock_bias_range_estimate
        )

        # ----------------------------------------------------
        # Résidu mesure - prédiction
        # ----------------------------------------------------

        residuals = (
            pseudoranges
            - predicted_pseudoranges
        )

        # ----------------------------------------------------
        # Jacobienne
        # ----------------------------------------------------

        H = build_pseudorange_jacobian(
            receiver_position=
                position_estimate,

            satellite_positions=
                satellite_positions
        )

        rank = np.linalg.matrix_rank(
            H
        )

        if rank < 4:
            raise RuntimeError(
                "La géométrie GNSS est dégénérée : "
                "rang(H) < 4."
            )

        # ----------------------------------------------------
        # Moindres carrés
        # ----------------------------------------------------

        correction, _, _, _ = (
            np.linalg.lstsq(
                H,
                residuals,
                rcond=None
            )
        )

        position_correction = (
            correction[0:3]
        )

        clock_bias_correction = (
            correction[3]
        )

        # ----------------------------------------------------
        # Mise à jour de l'état
        # ----------------------------------------------------

        position_estimate += (
            position_correction
        )

        clock_bias_range_estimate += (
            clock_bias_correction
        )

        # ----------------------------------------------------
        # Normes de correction
        # ----------------------------------------------------

        position_correction_norm = (
            np.linalg.norm(
                position_correction
            )
        )

        total_correction_norm = (
            np.linalg.norm(
                correction
            )
        )

        # ----------------------------------------------------
        # Historique
        # ----------------------------------------------------

        history.append({
            "iteration":
                iteration,

            "position_estimate":
                position_estimate.copy(),

            "clock_bias_range_estimate":
                clock_bias_range_estimate,

            "position_correction_norm":
                position_correction_norm,

            "clock_bias_correction":
                clock_bias_correction,

            "total_correction_norm":
                total_correction_norm
        })

        # ----------------------------------------------------
        # Critère de convergence
        # ----------------------------------------------------

        if (
            total_correction_norm
            < tolerance
        ):

            converged = True
            break

    # --------------------------------------------------------
    # Résidu final
    # --------------------------------------------------------

    (
        final_predicted_pseudoranges,
        _
    ) = predict_pseudoranges(
        receiver_position=
            position_estimate,

        satellite_positions=
            satellite_positions,

        clock_bias_range=
            clock_bias_range_estimate
    )

    final_residuals = (
        pseudoranges
        - final_predicted_pseudoranges
    )

    final_residual_rms = np.sqrt(
        np.mean(
            final_residuals**2
        )
    )

    # --------------------------------------------------------
    # Conditionnement final
    # --------------------------------------------------------

    final_H = build_pseudorange_jacobian(
        receiver_position=
            position_estimate,

        satellite_positions=
            satellite_positions
    )

    normal_matrix = (
        final_H.T
        @ final_H
    )

    condition_number = np.linalg.cond(
        normal_matrix
    )

    return {
        "position":
            position_estimate,

        "clock_bias_range":
            clock_bias_range_estimate,

        "converged":
            converged,

        "iterations":
            len(history),

        "history":
            history,

        "final_residuals":
            final_residuals,

        "final_residual_rms":
            final_residual_rms,

        "condition_number":
            condition_number
    }