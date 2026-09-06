import numpy as np

from scipy.linalg import expm

from src.attitude.quaternions import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_vector_to_quaternion,
    rotation_matrix_to_rotation_vector,
    propagate_quaternion_with_body_rate
)


# ============================================================
# AURORA
# Multiplicative Extended Kalman Filter (MEKF)
#
# Nominal state:
#
#     q_BI
#     b_g,B
#
# Error state:
#
#     delta_x =
#         [delta_theta,
#          delta_b_g]
#
# dimension = 6
#
#
# Quaternion convention:
#
#     BODY -> ECI
#
# Right multiplicative attitude error:
#
#     q_true =
#         q_nominal ⊗ delta_q
#
#
# Gyroscope:
#
#     omega_m =
#         omega_true
#         + b_g
#         + noise
# ============================================================


# ------------------------------------------------------------
# 1. MATRICE SKEW
# ------------------------------------------------------------

def skew_symmetric(
    vector
):
    """
    Retourne [v]_x telle que :

        [v]_x w = v x w
    """

    vector = np.asarray(
        vector,
        dtype=float
    )


    x, y, z = vector


    return np.array([
        [
            0.0,
            -z,
            y
        ],

        [
            z,
            0.0,
            -x
        ],

        [
            -y,
            x,
            0.0
        ]
    ])


# ------------------------------------------------------------
# 2. PREDICTION MEKF
# ------------------------------------------------------------

def predict_attitude_mekf(
    quaternion_body_to_eci,
    gyro_bias_body,
    covariance,
    gyro_measurement_body,
    dt,
    gyro_noise_std,
    gyro_bias_random_walk_density
):
    """
    Prediction MEKF.

    Vitesse angulaire estimee :

        omega_hat =
            omega_meas - b_hat

    Propagation :

        q^- =
            q^+ ⊗ delta_q(omega_hat dt)

    Dynamique lineaire de l'etat d'erreur :

        delta_theta_dot
            =
        -[omega_hat]_x delta_theta
        - delta_b

        delta_b_dot = 0
    """

    quaternion_body_to_eci = (
        normalize_quaternion(
            quaternion_body_to_eci
        )
    )


    gyro_bias_body = np.asarray(
        gyro_bias_body,
        dtype=float
    )


    covariance = np.asarray(
        covariance,
        dtype=float
    )


    gyro_measurement_body = np.asarray(
        gyro_measurement_body,
        dtype=float
    )


    if gyro_bias_body.shape != (3,):

        raise ValueError(
            "Le biais gyroscope doit avoir 3 composantes."
        )


    if gyro_measurement_body.shape != (3,):

        raise ValueError(
            "La mesure gyroscope doit avoir 3 composantes."
        )


    if covariance.shape != (6, 6):

        raise ValueError(
            "La covariance MEKF doit etre 6x6."
        )


    # --------------------------------------------------------
    # Mesure gyro corrigee du biais estime
    # --------------------------------------------------------

    corrected_angular_rate_body = (
        gyro_measurement_body
        -
        gyro_bias_body
    )


    # --------------------------------------------------------
    # Propagation quaternion nominal
    # --------------------------------------------------------

    predicted_quaternion = (
        propagate_quaternion_with_body_rate(
            quaternion_body_to_eci=
                quaternion_body_to_eci,

            angular_rate_body=
                corrected_angular_rate_body,

            dt=
                dt
        )
    )


    # --------------------------------------------------------
    # Jacobienne continue de l'etat d'erreur
    #
    # delta_x =
    #
    #     [delta_theta]
    #     [delta_b    ]
    # --------------------------------------------------------

    F = np.zeros(
        (
            6,
            6
        )
    )


    F[
        0:3,
        0:3
    ] = (
        -skew_symmetric(
            corrected_angular_rate_body
        )
    )


    F[
        0:3,
        3:6
    ] = (
        -np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Transition discrete
    #
    #     Phi = exp(F dt)
    # --------------------------------------------------------

    Phi = expm(
        F
        * dt
    )


    # --------------------------------------------------------
    # Bruit de processus
    # --------------------------------------------------------

    Q = np.zeros(
        (
            6,
            6
        )
    )


    # --------------------------------------------------------
    # Bruit gyro
    #
    # gyro_noise_std est ici traite comme un
    # bruit de vitesse angulaire par echantillon.
    #
    # L'erreur angulaire approximative sur dt :
    #
    #     sigma_theta ~ sigma_omega dt
    # --------------------------------------------------------

    Q[
        0:3,
        0:3
    ] = (
        gyro_noise_std**2
        * dt**2
        * np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Random walk biais gyro
    # --------------------------------------------------------

    Q[
        3:6,
        3:6
    ] = (
        gyro_bias_random_walk_density**2
        * dt
        * np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Propagation covariance
    # --------------------------------------------------------

    predicted_covariance = (
        Phi
        @ covariance
        @ Phi.T
        +
        Q
    )


    predicted_covariance = (
        0.5
        * (
            predicted_covariance
            +
            predicted_covariance.T
        )
    )


    return {
        "quaternion":
            predicted_quaternion,

        "gyro_bias":
            gyro_bias_body.copy(),

        "covariance":
            predicted_covariance,

        "corrected_angular_rate":
            corrected_angular_rate_body,

        "Phi":
            Phi,

        "Q":
            Q
    }


# ------------------------------------------------------------
# 3. UPDATE STAR TRACKER
# ------------------------------------------------------------

def update_attitude_mekf_with_star_tracker(
    predicted_quaternion,
    predicted_gyro_bias,
    predicted_covariance,
    star_tracker_quaternion,
    star_tracker_noise_std_rad
):
    """
    Correction absolue d'attitude par star tracker.

    Convention d'erreur multiplicative droite :

        R_true =
            R_nominal R_error

    Le residual est :

        R_res =
            R_pred^T R_meas

    Puis :

        innovation =
            Log(R_res)^vee

    Pour petits angles :

        innovation
            ~
        delta_theta + noise

    donc :

        H = [I  0]
    """

    predicted_quaternion = (
        normalize_quaternion(
            predicted_quaternion
        )
    )


    star_tracker_quaternion = (
        normalize_quaternion(
            star_tracker_quaternion
        )
    )


    predicted_gyro_bias = np.asarray(
        predicted_gyro_bias,
        dtype=float
    )


    predicted_covariance = np.asarray(
        predicted_covariance,
        dtype=float
    )


    if predicted_gyro_bias.shape != (3,):

        raise ValueError(
            "Le biais gyro predit doit avoir 3 composantes."
        )


    if predicted_covariance.shape != (6, 6):

        raise ValueError(
            "La covariance MEKF doit etre 6x6."
        )


    # --------------------------------------------------------
    # Quaternion -> matrices
    # --------------------------------------------------------

    predicted_rotation = (
        quaternion_to_rotation_matrix(
            predicted_quaternion
        )
    )


    measured_rotation = (
        quaternion_to_rotation_matrix(
            star_tracker_quaternion
        )
    )


    # --------------------------------------------------------
    # Residual multiplicatif droit
    # --------------------------------------------------------

    residual_rotation = (
        predicted_rotation.T
        @ measured_rotation
    )


    innovation = (
        rotation_matrix_to_rotation_vector(
            residual_rotation
        )
    )


    # --------------------------------------------------------
    # Matrice de mesure
    #
    #     z =
    #     [I 0] delta_x + noise
    # --------------------------------------------------------

    H = np.zeros(
        (
            3,
            6
        )
    )


    H[
        0:3,
        0:3
    ] = (
        np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Covariance star tracker
    # --------------------------------------------------------

    R_measurement = (
        star_tracker_noise_std_rad**2
        * np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Covariance innovation
    # --------------------------------------------------------

    innovation_covariance = (
        H
        @ predicted_covariance
        @ H.T
        +
        R_measurement
    )


    # --------------------------------------------------------
    # Gain de Kalman
    # --------------------------------------------------------

    kalman_gain = (
        predicted_covariance
        @ H.T
        @ np.linalg.inv(
            innovation_covariance
        )
    )


    # --------------------------------------------------------
    # Estimation de l'etat d'erreur
    # --------------------------------------------------------

    estimated_error_state = (
        kalman_gain
        @ innovation
    )


    estimated_attitude_error = (
        estimated_error_state[
            0:3
        ]
    )


    estimated_bias_error = (
        estimated_error_state[
            3:6
        ]
    )


    # --------------------------------------------------------
    # Injection multiplicative dans le quaternion
    #
    #     q+ =
    #         q- ⊗ delta_q
    # --------------------------------------------------------

    correction_quaternion = (
        rotation_vector_to_quaternion(
            estimated_attitude_error
        )
    )


    corrected_quaternion = (
        quaternion_multiply(
            predicted_quaternion,
            correction_quaternion
        )
    )


    corrected_quaternion = (
        normalize_quaternion(
            corrected_quaternion
        )
    )


    # --------------------------------------------------------
    # Correction biais gyro
    # --------------------------------------------------------

    corrected_gyro_bias = (
        predicted_gyro_bias
        +
        estimated_bias_error
    )


    # --------------------------------------------------------
    # Update covariance Joseph
    # --------------------------------------------------------

    identity = np.eye(
        6
    )


    correction_matrix = (
        identity
        -
        kalman_gain
        @ H
    )


    corrected_covariance = (
        correction_matrix
        @ predicted_covariance
        @ correction_matrix.T
        +
        kalman_gain
        @ R_measurement
        @ kalman_gain.T
    )


    # --------------------------------------------------------
    # Reset de l'etat d'erreur apres injection
    # --------------------------------------------------------

    reset_jacobian = np.eye(
        6
    )


    reset_jacobian[
        0:3,
        0:3
    ] = (
        np.eye(
            3
        )
        -
        0.5
        * skew_symmetric(
            estimated_attitude_error
        )
    )


    corrected_covariance = (
        reset_jacobian
        @ corrected_covariance
        @ reset_jacobian.T
    )


    corrected_covariance = (
        0.5
        * (
            corrected_covariance
            +
            corrected_covariance.T
        )
    )


    # --------------------------------------------------------
    # NIS
    #
    # Measurement dimension = 3
    #
    # Expected mean ~ 3
    # --------------------------------------------------------

    nis = (
        innovation.T
        @ np.linalg.solve(
            innovation_covariance,
            innovation
        )
    )


    return {
        "quaternion":
            corrected_quaternion,

        "gyro_bias":
            corrected_gyro_bias,

        "covariance":
            corrected_covariance,

        "innovation":
            innovation,

        "innovation_covariance":
            innovation_covariance,

        "kalman_gain":
            kalman_gain,

        "estimated_error_state":
            estimated_error_state,

        "nis":
            nis
    }