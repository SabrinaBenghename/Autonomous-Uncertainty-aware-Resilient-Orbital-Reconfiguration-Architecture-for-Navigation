import numpy as np

from src.dynamics.orbit import (
    MU_EARTH
)

from src.dynamics.perturbations import (
    j2_acceleration
)


# ============================================================
# AURORA
# Modèles dynamiques et prédiction EKF avancée
# ============================================================


# ------------------------------------------------------------
# 1. DYNAMIQUE DEUX CORPS
# ------------------------------------------------------------

def two_body_state_derivative(
    state
):
    """
    Dynamique orbitale deux-corps.

    Etat :
        [rx, ry, rz, vx, vy, vz]
    """

    position = np.asarray(
        state[0:3],
        dtype=float
    )

    velocity = np.asarray(
        state[3:6],
        dtype=float
    )

    radius = np.linalg.norm(
        position
    )

    acceleration = (
        -MU_EARTH
        * position
        / radius**3
    )

    return np.concatenate(
        (
            velocity,
            acceleration
        )
    )


# ------------------------------------------------------------
# 2. DYNAMIQUE DEUX CORPS + J2
# ------------------------------------------------------------

def j2_state_derivative(
    state
):
    """
    Dynamique orbitale incluant :
        - gravité centrale
        - perturbation J2
    """

    position = np.asarray(
        state[0:3],
        dtype=float
    )

    velocity = np.asarray(
        state[3:6],
        dtype=float
    )

    radius = np.linalg.norm(
        position
    )

    central_acceleration = (
        -MU_EARTH
        * position
        / radius**3
    )

    perturbation_acceleration = (
        j2_acceleration(
            position
        )
    )

    total_acceleration = (
        central_acceleration
        + perturbation_acceleration
    )

    return np.concatenate(
        (
            velocity,
            total_acceleration
        )
    )


# ------------------------------------------------------------
# 3. PROPAGATION RK4 DE L'ETAT
# ------------------------------------------------------------

def rk4_propagate_custom(
    state,
    dt,
    dynamics_function
):
    """
    Propagation RK4 d'un état orbital.
    """

    k1 = dynamics_function(
        state
    )

    k2 = dynamics_function(
        state
        + 0.5 * dt * k1
    )

    k3 = dynamics_function(
        state
        + 0.5 * dt * k2
    )

    k4 = dynamics_function(
        state
        + dt * k3
    )

    propagated_state = (
        state
        + dt
        / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    return propagated_state


# ------------------------------------------------------------
# 4. JACOBIENNE NUMERIQUE CONTINUE
# ------------------------------------------------------------

def numerical_dynamics_jacobian(
    state,
    dynamics_function,
    position_step=1.0,
    velocity_step=1e-3
):
    """
    Calcule :

        F = df / dx

    par différences centrées.
    """

    state = np.asarray(
        state,
        dtype=float
    )

    F = np.zeros(
        (
            6,
            6
        )
    )

    perturbation_steps = np.array([
        position_step,
        position_step,
        position_step,
        velocity_step,
        velocity_step,
        velocity_step
    ])


    for column in range(
        6
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
# 5. PROCESS NOISE D'ACCELERATION
# ------------------------------------------------------------

def build_discrete_acceleration_process_noise(
    dt,
    acceleration_sigma
):
    """
    Covariance Q correspondant à une
    accélération inconnue pendant dt.

    delta_r = 1/2 a dt²
    delta_v = a dt
    """

    if acceleration_sigma < 0.0:

        raise ValueError(
            "acceleration_sigma doit être >= 0."
        )


    G = np.zeros(
        (
            6,
            3
        )
    )


    G[
        0:3,
        :
    ] = (
        0.5
        * dt**2
        * np.eye(3)
    )


    G[
        3:6,
        :
    ] = (
        dt
        * np.eye(3)
    )


    acceleration_covariance = (
        acceleration_sigma**2
        * np.eye(3)
    )


    return (
        G
        @ acceleration_covariance
        @ G.T
    )


# ------------------------------------------------------------
# 6. ANCIENNE PREDICTION
# Phi ≈ I + F dt
# ------------------------------------------------------------

def ekf_predict_custom(
    state,
    covariance,
    dt,
    process_noise_covariance,
    dynamics_function
):
    """
    Ancienne prédiction conservée pour
    comparaison avec 007-E.
    """

    predicted_state = (
        rk4_propagate_custom(
            state=
                state,

            dt=
                dt,

            dynamics_function=
                dynamics_function
        )
    )


    F = numerical_dynamics_jacobian(
        state=
            state,

        dynamics_function=
            dynamics_function
    )


    Phi = (
        np.eye(6)
        + F * dt
    )


    predicted_covariance = (
        Phi
        @ covariance
        @ Phi.T
        + process_noise_covariance
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
# 7. PROPAGATION CONJOINTE ETAT + STM
# ------------------------------------------------------------

def rk4_propagate_state_and_stm(
    state,
    dt,
    dynamics_function
):
    """
    Propage simultanément :

        x_dot = f(x)

    et :

        Phi_dot = F(x) Phi

    avec :

        Phi(t0) = I

    Ceci donne une matrice de transition
    beaucoup plus cohérente avec la
    propagation non linéaire RK4.
    """

    state = np.asarray(
        state,
        dtype=float
    )


    phi_initial = np.eye(
        6
    )


    # --------------------------------------------------------
    # Etape RK4 numéro 1
    # --------------------------------------------------------

    k1_state = (
        dynamics_function(
            state
        )
    )


    F1 = numerical_dynamics_jacobian(
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
    # Etape RK4 numéro 2
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


    F2 = numerical_dynamics_jacobian(
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
    # Etape RK4 numéro 3
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


    F3 = numerical_dynamics_jacobian(
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
    # Etape RK4 numéro 4
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


    F4 = numerical_dynamics_jacobian(
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
# 8. PREDICTION EKF AVEC EQUATIONS VARIATIONNELLES
# ------------------------------------------------------------

def ekf_predict_variational(
    state,
    covariance,
    dt,
    process_noise_covariance,
    dynamics_function
):
    """
    Prediction EKF utilisant la STM issue
    des équations variationnelles :

        Phi_dot = F Phi

    au lieu de :

        Phi ≈ I + F dt
    """

    (
        predicted_state,
        Phi
    ) = rk4_propagate_state_and_stm(
        state=
            state,

        dt=
            dt,

        dynamics_function=
            dynamics_function
    )


    predicted_covariance = (
        Phi
        @ covariance
        @ Phi.T
        + process_noise_covariance
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