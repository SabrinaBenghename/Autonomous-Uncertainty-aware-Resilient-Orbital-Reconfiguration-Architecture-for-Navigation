import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH,
    orbital_period
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.perturbations import (
    j2_acceleration,
    propagate_orbit_with_j2
)

from src.navigation.orbit_ekf import (
    ekf_update_position
)

from src.navigation.orbit_ekf_models import (
    two_body_state_derivative,
    j2_state_derivative,
    build_discrete_acceleration_process_noise,
    ekf_predict_custom
)


# ============================================================
# AURORA
# Expérience 007-D
#
# Comparaison :
#
# A - Two-body, Q = 0
# B - Two-body, Q > 0
# C - J2 modélisé, Q = 0
#
# Vérité :
#     Two-body + J2
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH + 550_000.0
)

eccentricity = 0.01

inclination = np.deg2rad(
    97.6
)

raan = np.deg2rad(
    40.0
)

argument_of_periapsis = np.deg2rad(
    30.0
)

true_anomaly = np.deg2rad(
    25.0
)


# ------------------------------------------------------------
# 2. TEMPS
# ------------------------------------------------------------

period = orbital_period(
    semi_major_axis
)

simulation_duration = (
    2.0 * period
)

desired_dt = 10.0


number_of_epochs = (
    int(
        np.floor(
            simulation_duration
            / desired_dt
        )
    )
    + 1
)


time = np.linspace(
    0.0,
    simulation_duration,
    number_of_epochs
)


dt = (
    time[1]
    - time[0]
)


time_minutes = (
    time / 60.0
)


# ------------------------------------------------------------
# 3. COUPURE GNSS
# ------------------------------------------------------------

gnss_outage_start_minutes = 80.0

gnss_outage_end_minutes = 100.0


gnss_available = ~(
    (
        time_minutes
        >= gnss_outage_start_minutes
    )
    &
    (
        time_minutes
        <= gnss_outage_end_minutes
    )
)


# ------------------------------------------------------------
# 4. ETAT VRAI INITIAL
# ------------------------------------------------------------

true_initial_position, true_initial_velocity = (
    keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )
)


true_initial_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)


# ------------------------------------------------------------
# 5. VERITE J2
# ------------------------------------------------------------

truth_solution = (
    propagate_orbit_with_j2(
        initial_state=
            true_initial_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs
    )
)


truth_states = (
    truth_solution.y.T
)


# ------------------------------------------------------------
# 6. SYNCHRONISATION
# ------------------------------------------------------------

maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        - time
    )
)


# ------------------------------------------------------------
# 7. MESURES GNSS COMMUNES
# ------------------------------------------------------------

gnss_position_sigma = 3.0


R = (
    gnss_position_sigma**2
    * np.eye(3)
)


rng = np.random.default_rng(
    2026
)


gnss_measurements = (
    truth_states[:, 0:3]
    +
    rng.normal(
        loc=0.0,
        scale=gnss_position_sigma,
        size=(
            number_of_epochs,
            3
        )
    )
)


# ------------------------------------------------------------
# 8. ETAT INITIAL ESTIME
# ------------------------------------------------------------

initial_position_error = np.array([
    100.0,
    -80.0,
    60.0
])


initial_velocity_error = np.array([
    0.10,
    -0.08,
    0.05
])


estimated_initial_state = (
    true_initial_state
    +
    np.concatenate(
        (
            initial_position_error,
            initial_velocity_error
        )
    )
)


# ------------------------------------------------------------
# 9. COVARIANCE INITIALE
# ------------------------------------------------------------

initial_position_sigma = 100.0

initial_velocity_sigma = 0.10


P0 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2
])


# ------------------------------------------------------------
# 10. PROCESS NOISE
# ------------------------------------------------------------

# Cas B :
#
# La norme de J2 mesurée en 007-C
# était environ 1.6e-2 m/s².
#
# On utilise ici 2e-2 m/s² comme
# accélération inconnue équivalente.
#
# Ce n'est PAS encore un tuning final.

equivalent_acceleration_sigma = (
    2.0e-2
)


Q_zero = np.zeros(
    (
        6,
        6
    )
)


Q_acceleration = (
    build_discrete_acceleration_process_noise(
        dt=
            dt,

        acceleration_sigma=
            equivalent_acceleration_sigma
    )
)


# ------------------------------------------------------------
# 11. FONCTION D'EXECUTION D'UN FILTRE
# ------------------------------------------------------------

def run_filter(
    name,
    dynamics_function,
    process_noise_covariance
):

    estimated_state = (
        estimated_initial_state.copy()
    )

    covariance = (
        P0.copy()
    )


    position_errors = np.zeros(
        number_of_epochs
    )


    position_sigmas = np.zeros(
        number_of_epochs
    )


    nees_values = np.full(
        number_of_epochs,
        np.nan
    )


    nis_values = np.full(
        number_of_epochs,
        np.nan
    )


    # --------------------------------------------------------
    # Etat initial
    # --------------------------------------------------------

    initial_error = (
        estimated_state
        - truth_states[0]
    )


    position_errors[0] = np.linalg.norm(
        initial_error[
            0:3
        ]
    )


    position_sigmas[0] = np.sqrt(
        np.trace(
            covariance[
                0:3,
                0:3
            ]
        )
    )


    nees_values[0] = (
        initial_error.T
        @ np.linalg.solve(
            covariance,
            initial_error
        )
    )


    # --------------------------------------------------------
    # Boucle temporelle
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        (
            predicted_state,
            predicted_covariance
        ) = ekf_predict_custom(
            state=
                estimated_state,

            covariance=
                covariance,

            dt=
                dt,

            process_noise_covariance=
                process_noise_covariance,

            dynamics_function=
                dynamics_function
        )


        # ----------------------------------------------------
        # GNSS disponible
        # ----------------------------------------------------

        if gnss_available[
            index
        ]:

            update_result = (
                ekf_update_position(
                    predicted_state=
                        predicted_state,

                    predicted_covariance=
                        predicted_covariance,

                    position_measurement=
                        gnss_measurements[
                            index
                        ],

                    measurement_noise_covariance=
                        R
                )
            )


            innovation = (
                update_result[
                    "innovation"
                ]
            )


            innovation_covariance = (
                update_result[
                    "innovation_covariance"
                ]
            )


            nis_values[
                index
            ] = (
                innovation.T
                @ np.linalg.solve(
                    innovation_covariance,
                    innovation
                )
            )


            estimated_state = (
                update_result[
                    "state"
                ]
            )


            covariance = (
                update_result[
                    "covariance"
                ]
            )


        else:

            estimated_state = (
                predicted_state
            )

            covariance = (
                predicted_covariance
            )


        # ----------------------------------------------------
        # Erreur
        # ----------------------------------------------------

        estimation_error = (
            estimated_state
            - truth_states[
                index
            ]
        )


        position_errors[
            index
        ] = np.linalg.norm(
            estimation_error[
                0:3
            ]
        )


        position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                covariance[
                    0:3,
                    0:3
                ]
            )
        )


        try:

            nees_values[
                index
            ] = (
                estimation_error.T
                @ np.linalg.solve(
                    covariance,
                    estimation_error
                )
            )

        except np.linalg.LinAlgError:

            nees_values[
                index
            ] = np.nan


    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------

    outage_mask = (
        ~gnss_available
    )


    evaluation_mask = np.ones(
        number_of_epochs,
        dtype=bool
    )

    # On exclut l'état initial du RMSE
    # GNSS car il n'a pas encore été corrigé.

    evaluation_mask[0] = False


    available_evaluation_mask = (
        gnss_available
        & evaluation_mask
    )


    rmse_gnss = np.sqrt(
        np.mean(
            position_errors[
                available_evaluation_mask
            ]**2
        )
    )


    rmse_outage = np.sqrt(
        np.mean(
            position_errors[
                outage_mask
            ]**2
        )
    )


    outage_indices = np.where(
        outage_mask
    )[0]


    error_end_outage = (
        position_errors[
            outage_indices[-1]
        ]
    )


    sigma_end_outage = (
        position_sigmas[
            outage_indices[-1]
        ]
    )


    valid_nis = np.isfinite(
        nis_values
    )


    valid_nees = np.isfinite(
        nees_values
    )


    mean_nis = np.mean(
        nis_values[
            valid_nis
        ]
    )


    mean_nees = np.mean(
        nees_values[
            valid_nees
        ]
    )


    # --------------------------------------------------------
    # Première époque après outage
    # --------------------------------------------------------

    return_indices = np.where(
        time_minutes
        > gnss_outage_end_minutes
    )[0]


    first_return_index = (
        return_indices[0]
    )


    return_error = (
        position_errors[
            first_return_index
        ]
    )


    return {
        "name":
            name,

        "position_errors":
            position_errors,

        "position_sigmas":
            position_sigmas,

        "nees":
            nees_values,

        "nis":
            nis_values,

        "rmse_gnss":
            rmse_gnss,

        "rmse_outage":
            rmse_outage,

        "error_end_outage":
            error_end_outage,

        "sigma_end_outage":
            sigma_end_outage,

        "mean_nis":
            mean_nis,

        "mean_nees":
            mean_nees,

        "return_error":
            return_error
    }


# ------------------------------------------------------------
# 12. CAS A
# TWO-BODY, Q = 0
# ------------------------------------------------------------

result_two_body_q0 = (
    run_filter(
        name=
            "Two-body, Q=0",

        dynamics_function=
            two_body_state_derivative,

        process_noise_covariance=
            Q_zero
    )
)


# ------------------------------------------------------------
# 13. CAS B
# TWO-BODY + PROCESS NOISE
# ------------------------------------------------------------

result_two_body_q = (
    run_filter(
        name=
            "Two-body + Q",

        dynamics_function=
            two_body_state_derivative,

        process_noise_covariance=
            Q_acceleration
    )
)


# ------------------------------------------------------------
# 14. CAS C
# MODELE J2
# ------------------------------------------------------------

result_j2 = (
    run_filter(
        name=
            "J2 model",

        dynamics_function=
            j2_state_derivative,

        process_noise_covariance=
            Q_zero
    )
)


results = [
    result_two_body_q0,
    result_two_body_q,
    result_j2
]


# ------------------------------------------------------------
# 15. INFORMATIONS PHYSIQUES
# ------------------------------------------------------------

initial_j2_norm = np.linalg.norm(
    j2_acceleration(
        true_initial_position
    )
)


# ------------------------------------------------------------
# 16. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "======================================================================================"
)

print(
    "AURORA — Expérience 007-D"
)

print(
    "Process noise vs fidélité du modèle dynamique"
)

print(
    "======================================================================================"
)


print(
    f"Erreur synchronisation temporelle : "
    f"{maximum_time_error:.6e} s"
)


print(
    f"Norme J2 initiale : "
    f"{initial_j2_norm:.6e} m/s^2"
)


print(
    f"Sigma accélération du cas Q : "
    f"{equivalent_acceleration_sigma:.6e} m/s^2"
)


print()


print(
    "Modèle             | RMSE GNSS | RMSE outage | "
    "Err fin | Sigma fin | NIS moy. | NEES moy. | Retour"
)

print(
    "                   |    [m]    |     [m]     | "
    "  [m]    |   [m]     |          |           |  [m]"
)

print(
    "--------------------------------------------------------------------------------------"
)


for result in results:

    print(
        f"{result['name']:<18} | "
        f"{result['rmse_gnss']:>9.3f} | "
        f"{result['rmse_outage']:>11.3f} | "
        f"{result['error_end_outage']:>7.3f} | "
        f"{result['sigma_end_outage']:>9.3f} | "
        f"{result['mean_nis']:>8.3f} | "
        f"{result['mean_nees']:>9.3f} | "
        f"{result['return_error']:>7.3f}"
    )


print(
    "======================================================================================"
)


print()

print(
    "Références statistiques :"
)

print(
    "NIS moyen attendu ≈ 3"
)

print(
    "NEES moyen attendu ≈ 6"
)


# ------------------------------------------------------------
# 17. FIGURE ERREUR DE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


for result in results:

    plt.plot(
        time_minutes,
        result[
            "position_errors"
        ],
        label=
            result[
                "name"
            ]
    )


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15,
    label="Coupure GNSS"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur de position 3D [m]"
)


plt.title(
    "AURORA — Effet du modèle dynamique sur la navigation"
)


plt.yscale(
    "log"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. FIGURE INCERTITUDE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


for result in results:

    plt.plot(
        time_minutes,
        result[
            "position_sigmas"
        ],
        label=
            result[
                "name"
            ]
    )


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Incertitude position 3D [m]"
)


plt.title(
    "AURORA — Covariance annoncée par les filtres"
)


plt.yscale(
    "log"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FIGURE NEES
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


for result in results:

    plt.plot(
        time_minutes,
        result[
            "nees"
        ],
        label=
            result[
                "name"
            ]
    )


plt.axhline(
    6.0,
    linestyle="--",
    label="NEES moyen théorique = 6"
)


plt.axvspan(
    gnss_outage_start_minutes,
    gnss_outage_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "NEES [-]"
)


plt.title(
    "AURORA — Cohérence selon le modèle dynamique"
)


plt.yscale(
    "log"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()