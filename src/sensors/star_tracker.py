import numpy as np

from src.attitude.quaternions import (
    normalize_quaternion,
    quaternion_multiply,
    rotation_vector_to_quaternion
)


# ============================================================
# AURORA
# Star tracker measurement model
#
# The star tracker provides an absolute attitude measurement.
#
# True attitude:
#
#     q_BI,true
#
# Measurement:
#
#     q_BI,meas =
#         q_BI,true ⊗ delta_q_noise
#
# where delta_q_noise is generated from a small
# Gaussian rotation vector.
#
#
# Convention:
#
#     q_BI : BODY -> ECI
# ============================================================


def simulate_star_tracker_measurement(
    true_quaternion_body_to_eci,
    attitude_noise_std_rad,
    rng
):
    """
    Simule une mesure absolue d'attitude par star tracker.

    Parameters
    ----------
    true_quaternion_body_to_eci : ndarray (4,)
        Quaternion vrai BODY -> ECI.

    attitude_noise_std_rad : float
        Ecart-type par composante du petit vecteur
        d'erreur angulaire [rad].

    rng : numpy.random.Generator

    Returns
    -------
    dict
    """

    true_quaternion = (
        normalize_quaternion(
            true_quaternion_body_to_eci
        )
    )


    # --------------------------------------------------------
    # Petit bruit angulaire 3D
    # --------------------------------------------------------

    noise_rotation_vector = rng.normal(
        loc=0.0,
        scale=attitude_noise_std_rad,
        size=3
    )


    noise_quaternion = (
        rotation_vector_to_quaternion(
            noise_rotation_vector
        )
    )


    # --------------------------------------------------------
    # Mesure absolue d'attitude
    #
    # Erreur multiplicative droite :
    #
    #     q_meas = q_true ⊗ delta_q
    # --------------------------------------------------------

    measured_quaternion = (
        quaternion_multiply(
            true_quaternion,
            noise_quaternion
        )
    )


    return {
        "measurement":
            normalize_quaternion(
                measured_quaternion
            ),

        "true_quaternion":
            true_quaternion,

        "noise_rotation_vector":
            noise_rotation_vector,

        "noise_angle":
            np.linalg.norm(
                noise_rotation_vector
            )
    }