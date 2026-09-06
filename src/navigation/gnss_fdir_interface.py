import numpy as np

from scipy.stats import chi2

from src.navigation.gnss_ekf_interface import (
    solve_gnss_least_squares
)


# ============================================================
# AURORA
# Interface FDI / FDIR pour la chaine GNSS -> EKF
#
# Fonctions :
#
# 1. Test global des residus post-fit
# 2. Isolation leave-one-out
# 3. Reconfiguration par rejet du satellite isole
# ============================================================


# ------------------------------------------------------------
# 1. TEST GLOBAL CHI2
# ------------------------------------------------------------

def global_postfit_residual_test(
    residuals,
    pseudorange_noise_std,
    number_of_estimated_parameters=4,
    confidence=0.99
):
    """
    Test global des residus GNSS.

    Statistique :

        T = r^T r / sigma_rho^2

    Sous les hypotheses nominales :

        T ~ Chi2(dof)

    avec :

        dof = N - p

    N :
        nombre de pseudodistances

    p :
        nombre de parametres estimes
        ici [x, y, z, b] => p = 4
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )


    if pseudorange_noise_std <= 0.0:

        raise ValueError(
            "pseudorange_noise_std doit etre > 0."
        )


    number_of_measurements = (
        len(
            residuals
        )
    )


    degrees_of_freedom = (
        number_of_measurements
        - number_of_estimated_parameters
    )


    if degrees_of_freedom <= 0:

        return {
            "valid":
                False,

            "statistic":
                np.nan,

            "threshold":
                np.nan,

            "degrees_of_freedom":
                degrees_of_freedom,

            "detected":
                False
        }


    statistic = (
        residuals
        @ residuals
        / pseudorange_noise_std**2
    )


    threshold = chi2.ppf(
        confidence,
        degrees_of_freedom
    )


    detected = (
        statistic
        > threshold
    )


    return {
        "valid":
            True,

        "statistic":
            statistic,

        "threshold":
            threshold,

        "degrees_of_freedom":
            degrees_of_freedom,

        "detected":
            detected
    }


# ------------------------------------------------------------
# 2. ISOLATION LEAVE-ONE-OUT
# ------------------------------------------------------------

def isolate_fault_leave_one_out(
    satellite_positions,
    pseudoranges,
    satellite_ids,
    initial_position,
    initial_clock_bias_meters,
    pseudorange_noise_std,
    confidence=0.99
):
    """
    Retire chaque satellite successivement.

    Pour chaque hypothese :

        satellite i retire
            ↓
        nouvelle solution LS
            ↓
        nouveau test global
            ↓
        score = T / dof

    Le satellite dont le retrait produit
    le plus petit score est considere comme
    le candidat fautif.
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
        len(
            satellite_positions
        )
    )


    # Il faut garder suffisamment de redondance
    # apres suppression :
    #
    # N - 1 > 4
    #
    # pour avoir au moins un degre de liberte.

    if number_of_satellites < 6:

        return {
            "success":
                False,

            "reason":
                "Redondance insuffisante pour isolation robuste."
        }


    best_candidate = (
        None
    )


    for excluded_index in range(
        number_of_satellites
    ):

        keep_mask = np.ones(
            number_of_satellites,
            dtype=bool
        )


        keep_mask[
            excluded_index
        ] = False


        reduced_positions = (
            satellite_positions[
                keep_mask
            ]
        )


        reduced_pseudoranges = (
            pseudoranges[
                keep_mask
            ]
        )


        reduced_ids = [
            satellite_ids[index]
            for index in range(
                number_of_satellites
            )
            if keep_mask[index]
        ]


        try:

            reduced_solution = (
                solve_gnss_least_squares(
                    satellite_positions=
                        reduced_positions,

                    pseudoranges=
                        reduced_pseudoranges,

                    initial_position=
                        initial_position,

                    initial_clock_bias_meters=
                        initial_clock_bias_meters
                )
            )


            if not reduced_solution[
                "converged"
            ]:

                continue


            reduced_test = (
                global_postfit_residual_test(
                    residuals=
                        reduced_solution[
                            "residuals"
                        ],

                    pseudorange_noise_std=
                        pseudorange_noise_std,

                    number_of_estimated_parameters=
                        4,

                    confidence=
                        confidence
                )
            )


            if not reduced_test[
                "valid"
            ]:

                continue


            score = (
                reduced_test[
                    "statistic"
                ]
                /
                reduced_test[
                    "degrees_of_freedom"
                ]
            )


            candidate = {
                "excluded_index":
                    excluded_index,

                "excluded_id":
                    satellite_ids[
                        excluded_index
                    ],

                "score":
                    score,

                "solution":
                    reduced_solution,

                "test":
                    reduced_test,

                "satellite_positions":
                    reduced_positions,

                "pseudoranges":
                    reduced_pseudoranges,

                "satellite_ids":
                    reduced_ids
            }


            if (
                best_candidate is None
                or
                score
                <
                best_candidate[
                    "score"
                ]
            ):

                best_candidate = (
                    candidate
                )


        except (
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError
        ):

            continue


    if best_candidate is None:

        return {
            "success":
                False,

            "reason":
                "Aucune hypothese leave-one-out exploitable."
        }


    return {
        "success":
            True,

        **best_candidate
    }


# ------------------------------------------------------------
# 3. CHAINE FDIR COMPLETE
# ------------------------------------------------------------

def solve_gnss_with_fdir(
    satellite_positions,
    pseudoranges,
    satellite_ids,
    initial_position,
    initial_clock_bias_meters,
    pseudorange_noise_std,
    confidence=0.99
):
    """
    Chaine :

        LS avec tous les satellites
                ↓
        test global Chi2
                ↓
        si sain :
            accepter
                ↓
        si anomalie :
            isolation leave-one-out
                ↓
        si solution reduite saine :
            rejeter satellite
            accepter solution reconfiguree
                ↓
        sinon :
            ne pas fournir de mesure a l'EKF
    """

    # --------------------------------------------------------
    # Solution complete
    # --------------------------------------------------------

    full_solution = (
        solve_gnss_least_squares(
            satellite_positions=
                satellite_positions,

            pseudoranges=
                pseudoranges,

            initial_position=
                initial_position,

            initial_clock_bias_meters=
                initial_clock_bias_meters
        )
    )


    if not full_solution[
        "converged"
    ]:

        return {
            "measurement_accepted":
                False,

            "fault_detected":
                False,

            "reconfigured":
                False,

            "reason":
                "Solution GNSS complete non convergee."
        }


    # --------------------------------------------------------
    # Test global
    # --------------------------------------------------------

    full_test = (
        global_postfit_residual_test(
            residuals=
                full_solution[
                    "residuals"
                ],

            pseudorange_noise_std=
                pseudorange_noise_std,

            number_of_estimated_parameters=
                4,

            confidence=
                confidence
        )
    )


    if not full_test[
        "valid"
    ]:

        return {
            "measurement_accepted":
                False,

            "fault_detected":
                False,

            "reconfigured":
                False,

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "reason":
                "Test global non valide."
        }


    # --------------------------------------------------------
    # Aucun defaut detecte
    # --------------------------------------------------------

    if not full_test[
        "detected"
    ]:

        return {
            "measurement_accepted":
                True,

            "fault_detected":
                False,

            "reconfigured":
                False,

            "isolated_id":
                None,

            "solution":
                full_solution,

            "satellite_positions":
                np.asarray(
                    satellite_positions
                ),

            "pseudoranges":
                np.asarray(
                    pseudoranges
                ),

            "satellite_ids":
                list(
                    satellite_ids
                ),

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "accepted_test":
                full_test
        }


    # --------------------------------------------------------
    # Defaut detecte -> isolation
    # --------------------------------------------------------

    isolation_result = (
        isolate_fault_leave_one_out(
            satellite_positions=
                satellite_positions,

            pseudoranges=
                pseudoranges,

            satellite_ids=
                satellite_ids,

            initial_position=
                initial_position,

            initial_clock_bias_meters=
                initial_clock_bias_meters,

            pseudorange_noise_std=
                pseudorange_noise_std,

            confidence=
                confidence
        )
    )


    if not isolation_result[
        "success"
    ]:

        return {
            "measurement_accepted":
                False,

            "fault_detected":
                True,

            "reconfigured":
                False,

            "isolated_id":
                None,

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "reason":
                isolation_result[
                    "reason"
                ]
        }


    # --------------------------------------------------------
    # Verification apres rejet
    # --------------------------------------------------------

    reduced_test = (
        isolation_result[
            "test"
        ]
    )


    if reduced_test[
        "detected"
    ]:

        return {
            "measurement_accepted":
                False,

            "fault_detected":
                True,

            "reconfigured":
                False,

            "isolated_id":
                isolation_result[
                    "excluded_id"
                ],

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "accepted_test":
                reduced_test,

            "reason":
                "Anomalie toujours presente apres isolation."
        }


    # --------------------------------------------------------
    # Reconfiguration acceptee
    # --------------------------------------------------------

    return {
        "measurement_accepted":
            True,

        "fault_detected":
            True,

        "reconfigured":
            True,

        "isolated_id":
            isolation_result[
                "excluded_id"
            ],

        "solution":
            isolation_result[
                "solution"
            ],

        "satellite_positions":
            isolation_result[
                "satellite_positions"
            ],

        "pseudoranges":
            isolation_result[
                "pseudoranges"
            ],

        "satellite_ids":
            isolation_result[
                "satellite_ids"
            ],

        "full_solution":
            full_solution,

        "full_test":
            full_test,

        "accepted_test":
            reduced_test
    }