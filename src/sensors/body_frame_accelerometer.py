import numpy as np

from src.attitude.reference_attitude import (
    body_to_eci,
    eci_to_body
)


# ============================================================
# AURORA
# Accelerometre exprime dans le repere corps
#
# Modele :
#
#     f_m,B =
#         f_true,B
#         + b_B
#         + n_B
#
# avec :
#
#     f_true,B = R_BI^T f_true,I
#
# ============================================================


def simulate_body_frame_accelerometer_measurement(
    true_specific_force_eci,
    rotation_body_to_eci,
    bias_body,
    noise_std,
    rng
):
    """
    Simule une mesure accelerometre dans
    le repere corps du satellite.

    Parameters
    ----------
    true_specific_force_eci : ndarray (3,)
        Force specifique vraie en ECI [m/s^2].

    rotation_body_to_eci : ndarray (3,3)
        Matrice BODY -> ECI.

    bias_body : ndarray (3,)
        Biais accelerometre constant dans
        les axes physiques du capteur.

    noise_std : float
        Ecart-type du bruit sur chaque axe
        [m/s^2].

    rng : numpy random generator

    Returns
    -------
    dict
    """

    true_specific_force_eci = np.asarray(
        true_specific_force_eci,
        dtype=float
    )


    bias_body = np.asarray(
        bias_body,
        dtype=float
    )


    # --------------------------------------------------------
    # Force vraie vue par le capteur
    # --------------------------------------------------------

    true_specific_force_body = (
        eci_to_body(
            vector_eci=
                true_specific_force_eci,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    # --------------------------------------------------------
    # Bruit capteur dans les axes corps
    # --------------------------------------------------------

    noise_body = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=3
    )


    # --------------------------------------------------------
    # Mesure physique
    # --------------------------------------------------------

    measurement_body = (
        true_specific_force_body
        + bias_body
        + noise_body
    )


    # --------------------------------------------------------
    # Ce que donnerait cette mesure une fois
    # transformee vers ECI avec une attitude parfaite
    # --------------------------------------------------------

    measurement_eci = (
        body_to_eci(
            vector_body=
                measurement_body,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    bias_eci = (
        body_to_eci(
            vector_body=
                bias_body,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    noise_eci = (
        body_to_eci(
            vector_body=
                noise_body,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    return {
        "measurement_body":
            measurement_body,

        "measurement_eci":
            measurement_eci,

        "true_specific_force_body":
            true_specific_force_body,

        "true_specific_force_eci":
            true_specific_force_eci,

        "bias_body":
            bias_body,

        "bias_eci":
            bias_eci,

        "noise_body":
            noise_body,

        "noise_eci":
            noise_eci
    }