import numpy as np

from scipy.stats import chi2


# ============================================================
# AURORA
# GNSS anomaly feature engineering
#
# Phase 15-A
#
# Objectif :
#
# Transformer les diagnostics GNSS / FDIR deja existants
# en vecteur de caracteristiques utilisable par un
# detecteur data-driven.
#
#
# IMPORTANT :
#
# Les caracteristiques NE DOIVENT PAS utiliser :
#
#   - le label de faute
#   - l'amplitude vraie de faute
#   - la position vraie du satellite
#   - l'information "fault injected"
#
#
# Elles doivent etre calculables en operation a partir de :
#
#   - solution GNSS
#   - residus post-fit
#   - test chi2
#   - geometrie
#   - prediction/prior navigation
#
# Les labels sont reserves a l'evaluation ML.
# ============================================================


# ------------------------------------------------------------
# 1. FEATURE LIST
# ------------------------------------------------------------

AI_FEATURE_COLUMNS = [
    "number_of_satellites",
    "degrees_of_freedom",
    "pdop",
    "chi2_statistic",
    "chi2_threshold",
    "chi2_ratio",
    "chi2_per_dof",
    "residual_rms_m",
    "residual_std_m",
    "residual_mean_abs_m",
    "residual_max_abs_m",
    "residual_peak_to_peak_m",
    "normalized_residual_rms",
    "normalized_residual_max",
    "residual_peak_over_rms",
    "prior_position_correction_m",
    "prior_clock_correction_m"
]


# ------------------------------------------------------------
# 2. SAFE SCALAR
# ------------------------------------------------------------

def _safe_float(
    value,
    default=np.nan
):

    try:

        result = float(
            value
        )

        if np.isfinite(
            result
        ):

            return result

    except (
        TypeError,
        ValueError
    ):

        pass

    return float(
        default
    )


# ------------------------------------------------------------
# 3. FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_gnss_anomaly_features(
    fdir_result,
    number_of_satellites,
    pseudorange_noise_std,
    confidence,
    pdop,
    prior_position,
    prior_clock_bias_meters
):
    """
    Extrait les caracteristiques instantanees GNSS/FDIR.

    Parameters
    ----------
    fdir_result : dict
        Sortie de solve_gnss_with_fdir_transmit_time.

    number_of_satellites : int
        Nombre de pseudodistances utilisees dans le test global.

    pseudorange_noise_std : float
        Sigma suppose des pseudodistances [m].

    confidence : float
        Niveau de confiance du test chi2.

    pdop : float
        PDOP calcule a partir de la geometrie GNSS.

    prior_position : array-like
        Position de prediction / prior utilisee pour initialiser
        la solution GNSS.

    prior_clock_bias_meters : float
        Estimation prior du biais horloge [m].


    Returns
    -------
    features : dict or None
        None si les diagnostics necessaires ne sont pas disponibles.
    """

    if fdir_result is None:

        return None


    # --------------------------------------------------------
    # Full solution / full test
    #
    # On veut TOUJOURS les diagnostics avant reconfiguration.
    #
    # C'est important :
    #
    # une faute detectee puis correctement rejetee reste
    # une anomalie dans le signal brut.
    # --------------------------------------------------------

    full_solution = (
        fdir_result.get(
            "full_solution",
            None
        )
    )

    full_test = (
        fdir_result.get(
            "full_test",
            None
        )
    )


    # Cas nominal :
    # la solution acceptee est aussi la full solution.
    if full_solution is None:

        full_solution = (
            fdir_result.get(
                "solution",
                None
            )
        )


    if full_test is None:

        full_test = (
            fdir_result.get(
                "accepted_test",
                None
            )
        )


    if (
        full_solution is None
        or
        full_test is None
    ):

        return None


    if not full_solution.get(
        "converged",
        False
    ):

        return None


    if not full_test.get(
        "valid",
        False
    ):

        return None


    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    residuals = np.asarray(
        full_solution[
            "residuals"
        ],
        dtype=float
    )


    if (
        residuals.ndim
        != 1
        or
        residuals.size
        == 0
        or
        not np.all(
            np.isfinite(
                residuals
            )
        )
    ):

        return None


    # --------------------------------------------------------
    # Degrees of freedom
    # --------------------------------------------------------

    degrees_of_freedom = int(
        full_test.get(
            "degrees_of_freedom",
            number_of_satellites - 4
        )
    )


    if degrees_of_freedom <= 0:

        return None


    # --------------------------------------------------------
    # Chi-square statistics
    # --------------------------------------------------------

    chi2_statistic = _safe_float(
        full_test.get(
            "statistic",
            np.nan
        )
    )


    if not np.isfinite(
        chi2_statistic
    ):

        return None


    chi2_threshold = float(
        chi2.ppf(
            confidence,
            df=
                degrees_of_freedom
        )
    )


    chi2_ratio = (
        chi2_statistic
        /
        chi2_threshold
    )


    chi2_per_dof = (
        chi2_statistic
        /
        degrees_of_freedom
    )


    # --------------------------------------------------------
    # Residual features
    # --------------------------------------------------------

    residual_rms = np.sqrt(
        np.mean(
            residuals**2
        )
    )


    residual_std = np.std(
        residuals
    )


    residual_mean_abs = np.mean(
        np.abs(
            residuals
        )
    )


    residual_max_abs = np.max(
        np.abs(
            residuals
        )
    )


    residual_peak_to_peak = (
        np.max(
            residuals
        )
        -
        np.min(
            residuals
        )
    )


    normalized_residual_rms = (
        residual_rms
        /
        pseudorange_noise_std
    )


    normalized_residual_max = (
        residual_max_abs
        /
        pseudorange_noise_std
    )


    residual_peak_over_rms = (
        residual_max_abs
        /
        max(
            residual_rms,
            1.0e-12
        )
    )


    # --------------------------------------------------------
    # Prior-to-solution corrections
    #
    # Ces deux caracteristiques sont operationnelles :
    #
    # elles comparent la solution GNSS avec la prediction
    # disponible avant l'update.
    #
    # Elles n'utilisent PAS la verite.
    # --------------------------------------------------------

    estimated_position = np.asarray(
        full_solution[
            "position"
        ],
        dtype=float
    )


    prior_position = np.asarray(
        prior_position,
        dtype=float
    )


    prior_position_correction = np.linalg.norm(
        estimated_position
        -
        prior_position
    )


    estimated_clock_bias = _safe_float(
        full_solution.get(
            "clock_bias_meters",
            np.nan
        )
    )


    if not np.isfinite(
        estimated_clock_bias
    ):

        return None


    prior_clock_correction = abs(
        estimated_clock_bias
        -
        float(
            prior_clock_bias_meters
        )
    )


    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = {
        "number_of_satellites":
            float(
                number_of_satellites
            ),

        "degrees_of_freedom":
            float(
                degrees_of_freedom
            ),

        "pdop":
            float(
                pdop
            ),

        "chi2_statistic":
            float(
                chi2_statistic
            ),

        "chi2_threshold":
            float(
                chi2_threshold
            ),

        "chi2_ratio":
            float(
                chi2_ratio
            ),

        "chi2_per_dof":
            float(
                chi2_per_dof
            ),

        "residual_rms_m":
            float(
                residual_rms
            ),

        "residual_std_m":
            float(
                residual_std
            ),

        "residual_mean_abs_m":
            float(
                residual_mean_abs
            ),

        "residual_max_abs_m":
            float(
                residual_max_abs
            ),

        "residual_peak_to_peak_m":
            float(
                residual_peak_to_peak
            ),

        "normalized_residual_rms":
            float(
                normalized_residual_rms
            ),

        "normalized_residual_max":
            float(
                normalized_residual_max
            ),

        "residual_peak_over_rms":
            float(
                residual_peak_over_rms
            ),

        "prior_position_correction_m":
            float(
                prior_position_correction
            ),

        "prior_clock_correction_m":
            float(
                prior_clock_correction
            )
    }


    # --------------------------------------------------------
    # Final validity check
    # --------------------------------------------------------

    feature_vector = np.array(
        [
            features[
                column
            ]
            for column in AI_FEATURE_COLUMNS
        ],
        dtype=float
    )


    if not np.all(
        np.isfinite(
            feature_vector
        )
    ):

        return None


    return features