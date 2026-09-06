import numpy as np

from scipy.stats import chi2

from src.navigation.gnss_positioning import (
    build_pseudorange_jacobian,
    solve_position_least_squares
)


# ============================================================
# AURORA
# Phase 6 - Fault Detection and Isolation GNSS
# ============================================================


# ------------------------------------------------------------
# 1. TEST GLOBAL SUR LES RESIDUS
# ------------------------------------------------------------

def global_residual_test(
    residuals,
    measurement_noise_std,
    degrees_of_freedom,
    confidence_level=0.99
):
    """
    Réalise un test global de cohérence
    sur les résidus GNSS.

    Hypothèse nominale :
        r ~ bruit gaussien cohérent
        avec sigma.

    Statistique :
        T = r^T r / sigma^2

    Sous l'hypothèse nominale :
        T ~ Chi2(dof)

    Parameters
    ----------
    residuals : ndarray
        Résidus de pseudorange [m].

    measurement_noise_std : float
        Ecart-type nominal des mesures [m].

    degrees_of_freedom : int
        Nombre de degrés de liberté.

    confidence_level : float
        Niveau de confiance du test.

    Returns
    -------
    dict
        Statistique, seuil et décision.
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    if measurement_noise_std <= 0.0:
        raise ValueError(
            "measurement_noise_std doit être > 0."
        )

    if degrees_of_freedom <= 0:
        return {
            "statistic": np.nan,
            "threshold": np.nan,
            "detected": False
        }

    statistic = (
        np.dot(
            residuals,
            residuals
        )
        /
        measurement_noise_std**2
    )

    threshold = chi2.ppf(
        confidence_level,
        degrees_of_freedom
    )

    detected = (
        statistic
        > threshold
    )

    return {
        "statistic":
            statistic,

        "threshold":
            threshold,

        "detected":
            detected
    }


# ------------------------------------------------------------
# 2. RESIDUS NORMALISES
# ------------------------------------------------------------

def compute_normalized_residuals(
    receiver_position,
    satellite_positions,
    residuals,
    measurement_noise_std
):
    """
    Calcule les résidus normalisés en tenant
    compte de la géométrie GNSS.

    Returns
    -------
    ndarray
        Résidus normalisés.
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    H = build_pseudorange_jacobian(
        receiver_position=
            receiver_position,

        satellite_positions=
            satellite_positions
    )

    number_of_measurements = (
        H.shape[0]
    )

    # Matrice de projection sur l'espace
    # des résidus :
    #
    # M = I - H H+
    #
    # H+ est la pseudo-inverse de H.

    residual_projection_matrix = (
        np.eye(
            number_of_measurements
        )
        -
        H
        @ np.linalg.pinv(
            H
        )
    )

    residual_variances = (
        measurement_noise_std**2
        * np.diag(
            residual_projection_matrix
        )
    )

    # Protection contre les valeurs
    # numériques quasi nulles

    minimum_variance = (
        1e-12
        * measurement_noise_std**2
    )

    residual_variances = np.maximum(
        residual_variances,
        minimum_variance
    )

    normalized_residuals = (
        residuals
        /
        np.sqrt(
            residual_variances
        )
    )

    return normalized_residuals


# ------------------------------------------------------------
# 3. ISOLATION LEAVE-ONE-OUT
# ------------------------------------------------------------

def isolate_fault_leave_one_out(
    satellite_positions,
    pseudoranges,
    satellite_ids,
    initial_position,
    initial_clock_bias_range,
    measurement_noise_std,
    confidence_level=0.99,
    max_iterations=20
):
    """
    Essaie de supprimer chaque satellite
    l'un après l'autre.

    Le satellite dont la suppression rend
    les mesures restantes les plus cohérentes
    est considéré comme le meilleur candidat
    à la faute.

    Returns
    -------
    dict
        Satellite suspect et scores.
    """

    satellite_positions = np.asarray(
        satellite_positions,
        dtype=float
    )

    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )

    number_of_satellites = (
        satellite_positions.shape[0]
    )

    if number_of_satellites <= 4:
        return {
            "suspect_index": None,
            "suspect_id": None,
            "scores": []
        }

    scores = []

    best_index = None
    best_score = np.inf

    for excluded_index in range(
        number_of_satellites
    ):

        mask = np.ones(
            number_of_satellites,
            dtype=bool
        )

        mask[
            excluded_index
        ] = False

        reduced_positions = (
            satellite_positions[
                mask
            ]
        )

        reduced_pseudoranges = (
            pseudoranges[
                mask
            ]
        )

        solution = solve_position_least_squares(
            satellite_positions=
                reduced_positions,

            pseudoranges=
                reduced_pseudoranges,

            initial_position=
                initial_position,

            initial_clock_bias_range=
                initial_clock_bias_range,

            tolerance=
                1e-4,

            max_iterations=
                max_iterations
        )

        H = build_pseudorange_jacobian(
            receiver_position=
                solution[
                    "position"
                ],

            satellite_positions=
                reduced_positions
        )

        rank = np.linalg.matrix_rank(
            H
        )

        degrees_of_freedom = (
            len(
                reduced_pseudoranges
            )
            - rank
        )

        test_result = global_residual_test(
            residuals=
                solution[
                    "final_residuals"
                ],

            measurement_noise_std=
                measurement_noise_std,

            degrees_of_freedom=
                degrees_of_freedom,

            confidence_level=
                confidence_level
        )

        if (
            degrees_of_freedom > 0
            and np.isfinite(
                test_result[
                    "statistic"
                ]
            )
        ):

            normalized_score = (
                test_result[
                    "statistic"
                ]
                /
                degrees_of_freedom
            )

        else:

            normalized_score = np.inf

        scores.append({
            "excluded_index":
                excluded_index,

            "excluded_id":
                satellite_ids[
                    excluded_index
                ],

            "statistic":
                test_result[
                    "statistic"
                ],

            "threshold":
                test_result[
                    "threshold"
                ],

            "normalized_score":
                normalized_score,

            "remaining_consistent":
                not test_result[
                    "detected"
                ]
        })

        if (
            normalized_score
            < best_score
        ):

            best_score = (
                normalized_score
            )

            best_index = (
                excluded_index
            )

    if best_index is None:

        return {
            "suspect_index": None,
            "suspect_id": None,
            "scores": scores
        }

    return {
        "suspect_index":
            best_index,

        "suspect_id":
            satellite_ids[
                best_index
            ],

        "scores":
            scores
    }