import numpy as np

from src.navigation.orbit_ekf_models import (
    j2_state_derivative
)


# ============================================================
# AURORA
# EKF orbital 9 états avec estimation
# du biais accéléromètre
#
# Etat :
#
# x = [
#       rx, ry, rz,
#       vx, vy, vz,
#       bax, bay, baz
#     ]^T
#
# ============================================================


# ------------------------------------------------------------
# 1. DYNAMIQUE AUGMENTEE
# ------------------------------------------------------------

def build_bias_augmented_dynamics(
    accelerometer_measurement_eci
):
    """
    Construit la dynamique 9D :

        r_dot = v

        v_dot =
            gravity
            + J2
            + a_measured
            - b_estimated

        b_dot = 0

    Parameters
    ----------
    accelerometer_measurement_eci : ndarray
        Mesure accéléromètre en ECI [m/s^2].

    Returns
    -------
    callable
        Fonction dynamique 9D.
    """

    accelerometer_measurement_eci = (
        np.asarray(
            accelerometer_measurement_eci,
            dtype=float
        )
    )


    def dynamics_function(
        augmented_state
    ):

        orbital_state = (
            augmented_state[
                0:6
            ]
        )

        estimated_bias = (
            augmented_state[
                6:9
            ]
        )


        orbital_derivative = (
            j2_state_derivative(
                orbital_state
            ).copy()
        )


        corrected_specific_force = (
            accelerometer_measurement_eci
            - estimated_bias
        )


        orbital_derivative[
            3:6
        ] += corrected_specific_force


        bias_derivative = np.zeros(
            3
        )


        return np.concatenate(
            (
                orbital_derivative,
                bias_derivative
            )
        )


    return dynamics_function


# ------------------------------------------------------------
# 2. JACOBIENNE NUMERIQUE 9D
# ------------------------------------------------------------

def numerical_augmented_jacobian(
    state,
    dynamics_function
):
    """
    Jacobienne F = df/dx de la dynamique
    augmentée 9D par différences centrées.
    """

    state = np.asarray(
        state,
        dtype=float
    )


    state_dimension = (
        len(
            state
        )
    )


    F = np.zeros(
        (
            state_dimension,
            state_dimension
        )
    )


    perturbation_steps = np.array([
        1.0,
        1.0,
        1.0,

        1e-3,
        1e-3,
        1e-3,

        1e-8,
        1e-8,
        1e-8
    ])


    for column in range(
        state_dimension
    ):

        step = (
            perturbation_steps[
                column
            ]
        )


        state_plus = (
            state.copy()
        )


        state_minus = (
            state.copy()
        )


        state_plus[
            column
        ] += step


        state_minus[
            column
        ] -= step


        derivative_plus = (
            dynamics_function(
                state_plus
            )
        )


        derivative_minus = (
            dynamics_function(
                state_minus
            )
        )


        F[
            :,
            column
        ] = (
            derivative_plus
            - derivative_minus
        ) / (
            2.0 * step
        )


    return F


# ------------------------------------------------------------
# 3. PROPAGATION ETAT + STM 9D
# ------------------------------------------------------------

def rk4_propagate_augmented_state_and_stm(
    state,
    dt,
    dynamics_function
):
    """
    Propage simultanément :

        x_dot = f(x)

    et :

        Phi_dot = F(x) Phi

    pour l'état 9D.
    """

    state = np.asarray(
        state,
        dtype=float
    )


    state_dimension = (
        len(
            state
        )
    )


    phi_initial = np.eye(
        state_dimension
    )


    # --------------------------------------------------------
    # RK4 - étape 1
    # --------------------------------------------------------

    k1_state = (
        dynamics_function(
            state
        )
    )


    F1 = numerical_augmented_jacobian(
        state=
            state,

        dynamics_function=
            dynamics_function
    )


    k1_phi = (
        F1
        @ phi_initial
    )


    # --------------------------------------------------------
    # RK4 - étape 2
    # --------------------------------------------------------

    state_2 = (
        state
        + 0.5
        * dt
        * k1_state
    )


    phi_2 = (
        phi_initial
        + 0.5
        * dt
        * k1_phi
    )


    k2_state = (
        dynamics_function(
            state_2
        )
    )


    F2 = numerical_augmented_jacobian(
        state=
            state_2,

        dynamics_function=
            dynamics_function
    )


    k2_phi = (
        F2
        @ phi_2
    )


    # --------------------------------------------------------
    # RK4 - étape 3
    # --------------------------------------------------------

    state_3 = (
        state
        + 0.5
        * dt
        * k2_state
    )


    phi_3 = (
        phi_initial
        + 0.5
        * dt
        * k2_phi
    )


    k3_state = (
        dynamics_function(
            state_3
        )
    )


    F3 = numerical_augmented_jacobian(
        state=
            state_3,

        dynamics_function=
            dynamics_function
    )


    k3_phi = (
        F3
        @ phi_3
    )


    # --------------------------------------------------------
    # RK4 - étape 4
    # --------------------------------------------------------

    state_4 = (
        state
        + dt
        * k3_state
    )


    phi_4 = (
        phi_initial
        + dt
        * k3_phi
    )


    k4_state = (
        dynamics_function(
            state_4
        )
    )


    F4 = numerical_augmented_jacobian(
        state=
            state_4,

        dynamics_function=
            dynamics_function
    )


    k4_phi = (
        F4
        @ phi_4
    )


    # --------------------------------------------------------
    # Etat final
    # --------------------------------------------------------

    propagated_state = (
        state
        + dt
        / 6.0
        * (
            k1_state
            + 2.0 * k2_state
            + 2.0 * k3_state
            + k4_state
        )
    )


    # --------------------------------------------------------
    # STM finale
    # --------------------------------------------------------

    propagated_phi = (
        phi_initial
        + dt
        / 6.0
        * (
            k1_phi
            + 2.0 * k2_phi
            + 2.0 * k3_phi
            + k4_phi
        )
    )


    return (
        propagated_state,
        propagated_phi
    )


# ------------------------------------------------------------
# 4. PROCESS NOISE 9D
# ------------------------------------------------------------

def build_augmented_process_noise(
    dt,
    accelerometer_noise_std,
    bias_random_walk_density
):
    """
    Construit Q pour l'état 9D.

    Deux sources :

    1. bruit de mesure accéléromètre
    2. marche aléatoire lente du biais

    bias_random_walk_density :
        unités [m/s^2 / sqrt(s)]
    """

    Q = np.zeros(
        (
            9,
            9
        )
    )


    # --------------------------------------------------------
    # Bruit d'accélération
    # --------------------------------------------------------

    G_acceleration = np.zeros(
        (
            9,
            3
        )
    )


    G_acceleration[
        0:3,
        :
    ] = (
        0.5
        * dt**2
        * np.eye(3)
    )


    G_acceleration[
        3:6,
        :
    ] = (
        dt
        * np.eye(3)
    )


    acceleration_covariance = (
        accelerometer_noise_std**2
        * np.eye(3)
    )


    Q_acceleration = (
        G_acceleration
        @ acceleration_covariance
        @ G_acceleration.T
    )


    # --------------------------------------------------------
    # Marche aléatoire du biais
    # --------------------------------------------------------

    bias_variance_increment = (
        bias_random_walk_density**2
        * dt
    )


    Q_bias = np.zeros(
        (
            9,
            9
        )
    )


    Q_bias[
        6:9,
        6:9
    ] = (
        bias_variance_increment
        * np.eye(3)
    )


    Q = (
        Q_acceleration
        + Q_bias
    )


    return Q


# ------------------------------------------------------------
# 5. PREDICTION EKF 9D
# ------------------------------------------------------------

def ekf_predict_with_bias_estimation(
    state,
    covariance,
    dt,
    accelerometer_measurement_eci,
    accelerometer_noise_std,
    bias_random_walk_density
):
    """
    Prediction EKF 9D avec estimation
    explicite du biais accéléromètre.
    """

    dynamics_function = (
        build_bias_augmented_dynamics(
            accelerometer_measurement_eci
        )
    )


    (
        predicted_state,
        Phi
    ) = rk4_propagate_augmented_state_and_stm(
        state=
            state,

        dt=
            dt,

        dynamics_function=
            dynamics_function
    )


    Q = build_augmented_process_noise(
        dt=
            dt,

        accelerometer_noise_std=
            accelerometer_noise_std,

        bias_random_walk_density=
            bias_random_walk_density
    )


    predicted_covariance = (
        Phi
        @ covariance
        @ Phi.T
        + Q
    )


    predicted_covariance = (
        0.5
        * (
            predicted_covariance
            + predicted_covariance.T
        )
    )


    return (
        predicted_state,
        predicted_covariance
    )


# ------------------------------------------------------------
# 6. CORRECTION POSITION GNSS 9D
# ------------------------------------------------------------

def ekf_update_position_augmented(
    predicted_state,
    predicted_covariance,
    position_measurement,
    measurement_noise_covariance
):
    """
    Correction GNSS du filtre 9D.

    Le GNSS mesure directement uniquement
    les trois composantes de position.
    """

    H = np.zeros(
        (
            3,
            9
        )
    )


    H[
        :,
        0:3
    ] = np.eye(
        3
    )


    predicted_measurement = (
        H
        @ predicted_state
    )


    innovation = (
        position_measurement
        - predicted_measurement
    )


    innovation_covariance = (
        H
        @ predicted_covariance
        @ H.T
        + measurement_noise_covariance
    )


    kalman_gain = (
        predicted_covariance
        @ H.T
        @ np.linalg.inv(
            innovation_covariance
        )
    )


    updated_state = (
        predicted_state
        + kalman_gain
        @ innovation
    )


    # --------------------------------------------------------
    # Joseph form
    # --------------------------------------------------------

    identity = np.eye(
        9
    )


    I_minus_KH = (
        identity
        - kalman_gain
        @ H
    )


    updated_covariance = (
        I_minus_KH
        @ predicted_covariance
        @ I_minus_KH.T
        +
        kalman_gain
        @ measurement_noise_covariance
        @ kalman_gain.T
    )


    updated_covariance = (
        0.5
        * (
            updated_covariance
            + updated_covariance.T
        )
    )


    return {
        "state":
            updated_state,

        "covariance":
            updated_covariance,

        "innovation":
            innovation,

        "innovation_covariance":
            innovation_covariance,

        "kalman_gain":
            kalman_gain
    }