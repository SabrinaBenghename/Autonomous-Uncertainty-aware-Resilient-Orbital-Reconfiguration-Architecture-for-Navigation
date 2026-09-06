import numpy as np

from src.navigation.gnss_fdir_interface import (
    global_postfit_residual_test
)

from src.navigation.gnss_signal_propagation import (
    solve_gnss_least_squares_transmit_time
)


# ============================================================
# AURORA
# FDIR GNSS avec temps de propagation du signal
#
# Le solveur utilise :
#
#     r_sat(t_tx)
#
# avec :
#
#     t_tx = t_rx - tau
# ============================================================


# ------------------------------------------------------------
# 1. ISOLATION LEAVE-ONE-OUT
# ------------------------------------------------------------

def isolate_fault_leave_one_out_transmit_time(
    satellite_ids,
    pseudoranges,
    reception_time_seconds,
    initial_position,
    initial_clock_bias_meters,
    pseudorange_noise_std,
    confidence=0.99
):
    """
    Isolation d'une faute GNSS par retrait successif
    de chaque satellite.

    Pour chaque hypothese :

        retirer satellite i
                ↓
        resoudre [x,y,z,b] avec t_tx
                ↓
        test global des residus
                ↓
        score = T / dof

    Le meilleur candidat est celui donnant
    le plus petit score.
    """

    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )

    satellite_ids = list(
        satellite_ids
    )


    number_of_satellites = len(
        satellite_ids
    )


    # On veut conserver au moins 5 mesures
    # apres exclusion :
    #
    # 5 mesures - 4 inconnues = 1 ddl.
    if number_of_satellites < 6:

        return {
            "success": False,
            "reason": (
                "Redondance insuffisante "
                "pour isolation leave-one-out."
            )
        }


    best_candidate = None


    for excluded_index in range(
        number_of_satellites
    ):

        reduced_ids = [
            satellite_ids[index]
            for index in range(
                number_of_satellites
            )
            if index != excluded_index
        ]


        reduced_pseudoranges = np.delete(
            pseudoranges,
            excluded_index
        )


        try:

            reduced_solution = (
                solve_gnss_least_squares_transmit_time(
                    satellite_ids=
                        reduced_ids,

                    pseudoranges=
                        reduced_pseudoranges,

                    reception_time_seconds=
                        reception_time_seconds,

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

                "satellite_ids":
                    reduced_ids,

                "pseudoranges":
                    reduced_pseudoranges
            }


            if (
                best_candidate is None
                or
                candidate[
                    "score"
                ]
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
            "success": False,
            "reason": (
                "Aucune hypothese leave-one-out exploitable."
            )
        }


    return {
        "success": True,
        **best_candidate
    }


# ------------------------------------------------------------
# 2. CHAINE FDIR COMPLETE
# ------------------------------------------------------------

def solve_gnss_with_fdir_transmit_time(
    satellite_ids,
    pseudoranges,
    reception_time_seconds,
    initial_position,
    initial_clock_bias_meters,
    pseudorange_noise_std,
    confidence=0.99
):
    """
    Pipeline :

        pseudoranges
            ↓
        LS avec t_tx
            ↓
        test global chi2
            ↓
        si nominal :
            accepter
            ↓
        si anomalie :
            leave-one-out
            ↓
        si satellite isole :
            rejeter
            ↓
        nouvelle solution GNSS
    """

    satellite_ids = list(
        satellite_ids
    )


    pseudoranges = np.asarray(
        pseudoranges,
        dtype=float
    )


    # --------------------------------------------------------
    # Solution avec tous les satellites
    # --------------------------------------------------------

    full_solution = (
        solve_gnss_least_squares_transmit_time(
            satellite_ids=
                satellite_ids,

            pseudoranges=
                pseudoranges,

            reception_time_seconds=
                reception_time_seconds,

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

            "isolated_id":
                None,

            "reason":
                "Solution GNSS non convergee."
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

            "isolated_id":
                None,

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "reason":
                "Test global non valide."
        }


    # --------------------------------------------------------
    # Cas nominal
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

            "satellite_ids":
                satellite_ids,

            "pseudoranges":
                pseudoranges,

            "full_solution":
                full_solution,

            "full_test":
                full_test,

            "accepted_test":
                full_test
        }


    # --------------------------------------------------------
    # Faute detectee -> isolation
    # --------------------------------------------------------

    isolation_result = (
        isolate_fault_leave_one_out_transmit_time(
            satellite_ids=
                satellite_ids,

            pseudoranges=
                pseudoranges,

            reception_time_seconds=
                reception_time_seconds,

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


    reduced_test = (
        isolation_result[
            "test"
        ]
    )


    # --------------------------------------------------------
    # Toujours anormal apres rejet
    # --------------------------------------------------------

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
                (
                    "Anomalie toujours presente "
                    "apres exclusion."
                )
        }


    # --------------------------------------------------------
    # Reconfiguration validee
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

        "satellite_ids":
            isolation_result[
                "satellite_ids"
            ],

        "pseudoranges":
            isolation_result[
                "pseudoranges"
            ],

        "full_solution":
            full_solution,

        "full_test":
            full_test,

        "accepted_test":
            reduced_test
    }