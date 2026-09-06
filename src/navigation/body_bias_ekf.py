import numpy as np

from src.navigation.accelerometer_bias_ekf import (
    build_bias_augmented_dynamics
)


# ============================================================
# AURORA
# EKF translationnel avec biais accelerometre
# exprime dans le repere BODY.
#
# Etat :
#
#     x = [r_I, v_I, b_B]
#
# ou :
#
#     r_I : position ECI
#     v_I : vitesse ECI
#     b_B : biais accelerometre dans BODY
#
#
# Modele :
#
#     r_dot = v
#
#     v_dot =
#         a_gravity+J2
#         + R_BI @ (f_m,B - b_B)
#
#     b_dot_B = 0
#
#
# Phase 10-B :
#
#     l'attitude R_BI est supposee parfaitement connue.
# ============================================================


# ------------------------------------------------------------
# 1. EVALUATION DE LA DYNAMIQUE PHASE-8
# ------------------------------------------------------------

def evaluate_phase8_augmented_dynamics(
    state_with_eci_bias,
    accelerometer_measurement_eci
):
    """
    Adaptateur vers la dynamique 9D deja validee
    en Phase 8.

    Dans accelerometer_bias_ekf.py :

        build_bias_augmented_dynamics(
            accelerometer_measurement_eci
        )

    construit une fonction de dynamique.

    On doit donc faire :

        dynamics = build_bias_augmented_dynamics(a_meas)

        derivative = dynamics(state)
    """

    state_with_eci_bias = np.asarray(
        state_with_eci_bias,
        dtype=float
    )

    accelerometer_measurement_eci = np.asarray(
        accelerometer_measurement_eci,
        dtype=float
    )


    # --------------------------------------------------------
    # Construction de la fonction de dynamique Phase-8
    # --------------------------------------------------------

    dynamics_function = (
        build_bias_augmented_dynamics(
            accelerometer_measurement_eci
        )
    )


    # --------------------------------------------------------
    # Evaluation de cette dynamique sur l'etat
    # --------------------------------------------------------

    derivative = (
        dynamics_function(
            state_with_eci_bias
        )
    )


    derivative = np.asarray(
        derivative,
        dtype=float
    )


    if derivative.shape != (9,):

        raise ValueError(
            "La dynamique Phase-8 doit retourner un vecteur 9D."
        )


    return derivative


# ------------------------------------------------------------
# 2. DYNAMIQUE AVEC BIAIS BODY
# ------------------------------------------------------------

def build_body_bias_augmented_dynamics(
    state,
    accelerometer_measurement_body,
    rotation_body_to_eci
):
    """
    Dynamique de :

        x = [r_ECI, v_ECI, b_BODY]

    La mesure accelerometre est fournie
    dans le repere BODY.

    Le biais est estime dans BODY.

    Pour calculer l'acceleration orbitale :

        f_m,I = R_BI @ f_m,B

        b_I = R_BI @ b_B

    puis :

        f_corr,I =
            f_m,I - b_I
    """

    state = np.asarray(
        state,
        dtype=float
    )

    accelerometer_measurement_body = np.asarray(
        accelerometer_measurement_body,
        dtype=float
    )

    rotation_body_to_eci = np.asarray(
        rotation_body_to_eci,
        dtype=float
    )


    # --------------------------------------------------------
    # Verification dimensions
    # --------------------------------------------------------

    if state.shape != (9,):

        raise ValueError(
            "L'etat BODY-bias doit avoir 9 composantes."
        )


    if accelerometer_measurement_body.shape != (3,):

        raise ValueError(
            "La mesure accelerometre BODY doit avoir 3 composantes."
        )


    if rotation_body_to_eci.shape != (3, 3):

        raise ValueError(
            "R_BI doit etre une matrice 3x3."
        )


    # --------------------------------------------------------
    # Biais estime dans BODY
    # --------------------------------------------------------

    estimated_bias_body = (
        state[
            6:9
        ]
    )


    # --------------------------------------------------------
    # Mesure BODY -> ECI
    # --------------------------------------------------------

    accelerometer_measurement_eci = (
        rotation_body_to_eci
        @ accelerometer_measurement_body
    )


    # --------------------------------------------------------
    # Biais BODY -> ECI
    # --------------------------------------------------------

    estimated_bias_eci = (
        rotation_body_to_eci
        @ estimated_bias_body
    )


    # --------------------------------------------------------
    # Etat temporaire compatible Phase-8
    #
    # Phase-8 utilise :
    #
    #     [r_I, v_I, b_I]
    #
    # Mais ceci n'est PAS notre etat filtre.
    #
    # Notre vrai etat reste :
    #
    #     [r_I, v_I, b_B]
    # --------------------------------------------------------

    temporary_eci_bias_state = np.concatenate(
        (
            state[
                0:6
            ],

            estimated_bias_eci
        )
    )


    # --------------------------------------------------------
    # Dynamique orbitale deja validee
    # --------------------------------------------------------

    phase8_derivative = (
        evaluate_phase8_augmented_dynamics(
            state_with_eci_bias=
                temporary_eci_bias_state,

            accelerometer_measurement_eci=
                accelerometer_measurement_eci
        )
    )


    derivative = np.zeros(
        9
    )


    # --------------------------------------------------------
    # Position + vitesse
    # --------------------------------------------------------

    derivative[
        0:6
    ] = (
        phase8_derivative[
            0:6
        ]
    )


    # --------------------------------------------------------
    # Biais constant dans BODY
    #
    #     db_B/dt = 0
    # --------------------------------------------------------

    derivative[
        6:9
    ] = (
        0.0
    )


    return derivative


# ------------------------------------------------------------
# 3. JACOBIENNE NUMERIQUE
# ------------------------------------------------------------

def numerical_body_bias_jacobian(
    state,
    accelerometer_measurement_body,
    rotation_body_to_eci
):
    """
    Calcule :

        F = df/dx

    par differences centrees.

    Etat :

        [r_I, v_I, b_B]

    R_BI est ici consideree connue et constante
    pendant l'intervalle de propagation.
    """

    state = np.asarray(
        state,
        dtype=float
    )


    F = np.zeros(
        (
            9,
            9
        )
    )


    # --------------------------------------------------------
    # Pas de differentiation
    # --------------------------------------------------------

    perturbation_steps = np.array([
        # Position [m]
        1.0,
        1.0,
        1.0,

        # Vitesse [m/s]
        1.0e-3,
        1.0e-3,
        1.0e-3,

        # Biais [m/s²]
        1.0e-8,
        1.0e-8,
        1.0e-8
    ])


    for column in range(
        9
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
        ] += (
            step
        )


        state_minus[
            column
        ] -= (
            step
        )


        dynamics_plus = (
            build_body_bias_augmented_dynamics(
                state=
                    state_plus,

                accelerometer_measurement_body=
                    accelerometer_measurement_body,

                rotation_body_to_eci=
                    rotation_body_to_eci
            )
        )


        dynamics_minus = (
            build_body_bias_augmented_dynamics(
                state=
                    state_minus,

                accelerometer_measurement_body=
                    accelerometer_measurement_body,

                rotation_body_to_eci=
                    rotation_body_to_eci
            )
        )


        F[
            :,
            column
        ] = (
            dynamics_plus
            -
            dynamics_minus
        ) / (
            2.0
            * step
        )


    return F


# ------------------------------------------------------------
# 4. PROPAGATION RK4 ETAT + STM
# ------------------------------------------------------------

def rk4_propagate_body_bias_state_and_stm(
    state,
    dt,
    accelerometer_measurement_body,
    rotation_body_to_eci
):
    """
    Integre simultanement :

        x_dot = f(x)

    et :

        Phi_dot = F Phi

    avec :

        Phi(t0) = I

    via Runge-Kutta ordre 4.
    """

    state = np.asarray(
        state,
        dtype=float
    )


    # --------------------------------------------------------
    # STM initiale
    # --------------------------------------------------------

    Phi_initial = np.eye(
        9
    )


    # --------------------------------------------------------
    # Etat augmente :
    #
    # 9 etats physiques
    # +
    # 81 coefficients de Phi
    #
    # = 90 variables
    # --------------------------------------------------------

    augmented_initial = np.concatenate(
        (
            state,

            Phi_initial.reshape(
                -1
            )
        )
    )


    # --------------------------------------------------------
    # Dynamique augmentee
    # --------------------------------------------------------

    def augmented_derivative(
        augmented_state
    ):

        current_state = (
            augmented_state[
                0:9
            ]
        )


        current_Phi = (
            augmented_state[
                9:
            ].reshape(
                (
                    9,
                    9
                )
            )
        )


        # ----------------------------------------------------
        # Etat
        # ----------------------------------------------------

        state_derivative = (
            build_body_bias_augmented_dynamics(
                state=
                    current_state,

                accelerometer_measurement_body=
                    accelerometer_measurement_body,

                rotation_body_to_eci=
                    rotation_body_to_eci
            )
        )


        # ----------------------------------------------------
        # Jacobienne
        # ----------------------------------------------------

        F = (
            numerical_body_bias_jacobian(
                state=
                    current_state,

                accelerometer_measurement_body=
                    accelerometer_measurement_body,

                rotation_body_to_eci=
                    rotation_body_to_eci
            )
        )


        # ----------------------------------------------------
        # Equation variationnelle
        #
        #     Phi_dot = F Phi
        # ----------------------------------------------------

        Phi_derivative = (
            F
            @ current_Phi
        )


        return np.concatenate(
            (
                state_derivative,

                Phi_derivative.reshape(
                    -1
                )
            )
        )


    # --------------------------------------------------------
    # RK4
    # --------------------------------------------------------

    k1 = (
        augmented_derivative(
            augmented_initial
        )
    )


    k2 = (
        augmented_derivative(
            augmented_initial
            +
            0.5
            * dt
            * k1
        )
    )


    k3 = (
        augmented_derivative(
            augmented_initial
            +
            0.5
            * dt
            * k2
        )
    )


    k4 = (
        augmented_derivative(
            augmented_initial
            +
            dt
            * k3
        )
    )


    augmented_final = (
        augmented_initial
        +
        dt
        / 6.0
        * (
            k1
            +
            2.0
            * k2
            +
            2.0
            * k3
            +
            k4
        )
    )


    propagated_state = (
        augmented_final[
            0:9
        ]
    )


    Phi = (
        augmented_final[
            9:
        ].reshape(
            (
                9,
                9
            )
        )
    )


    return (
        propagated_state,
        Phi
    )


# ------------------------------------------------------------
# 5. BRUIT DE PROCESSUS
# ------------------------------------------------------------

def build_body_bias_process_noise(
    dt,
    accelerometer_noise_std,
    bias_random_walk_density,
    rotation_body_to_eci
):
    """
    Covariance discrete de bruit de processus.

    Bruit capteur :

        n_B ~ N(0, sigma_a² I)

    Transformation vers ECI :

        n_I = R_BI n_B

    Approximation discrete :

                    [0.5 dt² R_BI]
        G_accel  =  [   dt    R_BI]
                    [       0       ]

    Puis :

        Q_a =
            G_accel
            Sigma_a
            G_accel^T

    Le biais BODY recoit en plus un
    petit random walk.
    """

    rotation_body_to_eci = np.asarray(
        rotation_body_to_eci,
        dtype=float
    )


    # --------------------------------------------------------
    # Matrice d'influence du bruit accelerometre
    # --------------------------------------------------------

    G_accelerometer = np.zeros(
        (
            9,
            3
        )
    )


    G_accelerometer[
        0:3,
        :
    ] = (
        0.5
        * dt**2
        * rotation_body_to_eci
    )


    G_accelerometer[
        3:6,
        :
    ] = (
        dt
        * rotation_body_to_eci
    )


    # --------------------------------------------------------
    # Covariance du bruit BODY
    # --------------------------------------------------------

    accelerometer_noise_covariance_body = (
        accelerometer_noise_std**2
        * np.eye(
            3
        )
    )


    # --------------------------------------------------------
    # Contribution accelerometre
    # --------------------------------------------------------

    Q_accelerometer = (
        G_accelerometer
        @ accelerometer_noise_covariance_body
        @ G_accelerometer.T
    )


    # --------------------------------------------------------
    # Random walk biais BODY
    # --------------------------------------------------------

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
        bias_random_walk_density**2
        * dt
        * np.eye(
            3
        )
    )


    Q = (
        Q_accelerometer
        +
        Q_bias
    )


    # --------------------------------------------------------
    # Symetrisation numerique
    # --------------------------------------------------------

    Q = (
        0.5
        * (
            Q
            +
            Q.T
        )
    )


    return Q


# ------------------------------------------------------------
# 6. PREDICTION EKF
# ------------------------------------------------------------

def ekf_predict_with_body_bias_estimation(
    state,
    covariance,
    dt,
    accelerometer_measurement_body,
    rotation_body_to_eci,
    accelerometer_noise_std,
    bias_random_walk_density
):
    """
    Prediction EKF pour :

        x = [r_I, v_I, b_B]
    """

    state = np.asarray(
        state,
        dtype=float
    )


    covariance = np.asarray(
        covariance,
        dtype=float
    )


    if state.shape != (9,):

        raise ValueError(
            "L'etat EKF BODY doit avoir 9 composantes."
        )


    if covariance.shape != (9, 9):

        raise ValueError(
            "La covariance EKF BODY doit etre 9x9."
        )


    # --------------------------------------------------------
    # Propagation moyenne + STM
    # --------------------------------------------------------

    (
        predicted_state,
        Phi
    ) = (
        rk4_propagate_body_bias_state_and_stm(
            state=
                state,

            dt=
                dt,

            accelerometer_measurement_body=
                accelerometer_measurement_body,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    # --------------------------------------------------------
    # Process noise
    # --------------------------------------------------------

    Q = (
        build_body_bias_process_noise(
            dt=
                dt,

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                bias_random_walk_density,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    # --------------------------------------------------------
    # Propagation covariance
    #
    #     P_k+1^- =
    #         Phi P_k^+ Phi^T + Q
    # --------------------------------------------------------

    predicted_covariance = (
        Phi
        @ covariance
        @ Phi.T
        + Q
    )


    # --------------------------------------------------------
    # Symetrisation numerique
    # --------------------------------------------------------

    predicted_covariance = (
        0.5
        * (
            predicted_covariance
            +
            predicted_covariance.T
        )
    )


    return (
        predicted_state,
        predicted_covariance
    )