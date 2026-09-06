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
    J2_EARTH,
    j2_acceleration,
    propagate_orbit_with_j2
)

from src.navigation.orbit_ekf import (
    ekf_predict,
    ekf_update_position
)


# ============================================================
# AURORA
# Expérience 007-C
#
# Vérité :
#   Two-body + J2
#
# EKF :
#   Two-body uniquement
#
# Objectif :
#   Étudier l'effet d'une erreur de modèle
#   pendant une coupure GNSS.
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
# 4. ETAT INITIAL VRAI
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
# 5. VERITE AVEC J2
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

maximum_time_grid_error = np.max(
    np.abs(
        truth_solution.t
        - time
    )
)


# ------------------------------------------------------------
# 7. MAGNITUDE INITIALE DE J2
# ------------------------------------------------------------

initial_j2_acceleration = (
    j2_acceleration(
        true_initial_position
    )
)


initial_j2_acceleration_norm = (
    np.linalg.norm(
        initial_j2_acceleration
    )
)


# ------------------------------------------------------------
# 8. MESURES GNSS
# ------------------------------------------------------------

gnss_position_sigma = 3.0


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


R = (
    gnss_position_sigma**2
    * np.eye(3)
)


# ------------------------------------------------------------
# 9. ETAT INITIAL ESTIME
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


estimated_state = (
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
# 10. COVARIANCE INITIALE
# ------------------------------------------------------------

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)


estimated_covariance = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2
])


# ------------------------------------------------------------
# 11. PROCESS NOISE
#
# IMPORTANT :
#
# Q = 0 volontairement.
#
# On veut montrer ce qui se passe lorsque
# le filtre croit son modèle deux-corps
# parfait alors que la vérité contient J2.
# ------------------------------------------------------------

Q = np.zeros(
    (
        6,
        6
    )
)


# ------------------------------------------------------------
# 12. STOCKAGE
# ------------------------------------------------------------

estimated_states = np.zeros(
    (
        number_of_epochs,
        6
    )
)


position_errors = np.zeros(
    number_of_epochs
)


velocity_errors = np.zeros(
    number_of_epochs
)


position_sigma_3d = np.zeros(
    number_of_epochs
)


nees_values = np.full(
    number_of_epochs,
    np.nan
)


innovation_norms = np.full(
    number_of_epochs,
    np.nan
)


estimated_states[0] = (
    estimated_state
)


# ------------------------------------------------------------
# 13. ETAT INITIAL
# ------------------------------------------------------------

initial_estimation_error = (
    estimated_state
    - truth_states[0]
)


position_errors[0] = np.linalg.norm(
    initial_estimation_error[
        0:3
    ]
)


velocity_errors[0] = np.linalg.norm(
    initial_estimation_error[
        3:6
    ]
)


position_sigma_3d[0] = np.sqrt(
    np.trace(
        estimated_covariance[
            0:3,
            0:3
        ]
    )
)


nees_values[0] = (
    initial_estimation_error.T
    @ np.linalg.solve(
        estimated_covariance,
        initial_estimation_error
    )
)


# ------------------------------------------------------------
# 14. BOUCLE EKF
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # --------------------------------------------------------
    # PREDICTION
    #
    # ATTENTION :
    # ekf_predict utilise encore le modèle
    # deux-corps, SANS J2.
    # --------------------------------------------------------

    (
        predicted_state,
        predicted_covariance
    ) = ekf_predict(
        state=
            estimated_state,

        covariance=
            estimated_covariance,

        dt=
            dt,

        process_noise_covariance=
            Q
    )


    # --------------------------------------------------------
    # CORRECTION GNSS
    # --------------------------------------------------------

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


        estimated_state = (
            update_result[
                "state"
            ]
        )


        estimated_covariance = (
            update_result[
                "covariance"
            ]
        )


        innovation_norms[
            index
        ] = np.linalg.norm(
            update_result[
                "innovation"
            ]
        )


    else:

        estimated_state = (
            predicted_state
        )

        estimated_covariance = (
            predicted_covariance
        )


    # --------------------------------------------------------
    # STOCKAGE
    # --------------------------------------------------------

    estimated_states[
        index
    ] = estimated_state


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


    velocity_errors[
        index
    ] = np.linalg.norm(
        estimation_error[
            3:6
        ]
    )


    position_sigma_3d[
        index
    ] = np.sqrt(
        np.trace(
            estimated_covariance[
                0:3,
                0:3
            ]
        )
    )


    # --------------------------------------------------------
    # NEES
    # --------------------------------------------------------

    try:

        nees_values[
            index
        ] = (
            estimation_error.T
            @ np.linalg.solve(
                estimated_covariance,
                estimation_error
            )
        )

    except np.linalg.LinAlgError:

        nees_values[
            index
        ] = np.nan


# ------------------------------------------------------------
# 15. MASQUES
# ------------------------------------------------------------

outage_mask = (
    ~gnss_available
)


available_mask = (
    gnss_available
)


valid_nees_mask = np.isfinite(
    nees_values
)


# ------------------------------------------------------------
# 16. STATISTIQUES GLOBALES
# ------------------------------------------------------------

global_position_rmse = np.sqrt(
    np.mean(
        position_errors**2
    )
)


gnss_available_rmse = np.sqrt(
    np.mean(
        position_errors[
            available_mask
        ]**2
    )
)


outage_rmse = np.sqrt(
    np.mean(
        position_errors[
            outage_mask
        ]**2
    )
)


# ------------------------------------------------------------
# 17. DEBUT / FIN DE COUPURE
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]


error_at_outage_start = (
    position_errors[
        outage_indices[0]
    ]
)


error_at_outage_end = (
    position_errors[
        outage_indices[-1]
    ]
)


maximum_outage_error = np.max(
    position_errors[
        outage_mask
    ]
)


sigma_at_outage_start = (
    position_sigma_3d[
        outage_indices[0]
    ]
)


sigma_at_outage_end = (
    position_sigma_3d[
        outage_indices[-1]
    ]
)


nees_at_outage_start = (
    nees_values[
        outage_indices[0]
    ]
)


nees_at_outage_end = (
    nees_values[
        outage_indices[-1]
    ]
)


# ------------------------------------------------------------
# 18. PREMIERE CORRECTION APRES COUPURE
# ------------------------------------------------------------

after_outage_indices = np.where(
    time_minutes
    > gnss_outage_end_minutes
)[0]


if len(
    after_outage_indices
) > 0:

    first_after_outage = (
        after_outage_indices[0]
    )


    first_return_error = (
        position_errors[
            first_after_outage
        ]
    )

else:

    first_return_error = (
        np.nan
    )


# ------------------------------------------------------------
# 19. NEES MOYEN
# ------------------------------------------------------------

mean_nees = np.mean(
    nees_values[
        valid_nees_mask
    ]
)


# ------------------------------------------------------------
# 20. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================"
)

print(
    "AURORA — Expérience 007-C"
)

print(
    "Erreur de modèle : vérité J2 / EKF deux-corps"
)

print(
    "============================================================"
)


print(
    f"Coefficient J2 : "
    f"{J2_EARTH:.10e}"
)


print(
    f"Norme accélération J2 initiale : "
    f"{initial_j2_acceleration_norm:.6e} m/s^2"
)


print(
    f"Pas temporel : "
    f"{dt:.9f} s"
)


print(
    f"Erreur synchronisation temporelle : "
    f"{maximum_time_grid_error:.6e} s"
)


print()


print(
    "----- PERFORMANCE GLOBALE -----"
)


print(
    f"RMSE position global : "
    f"{global_position_rmse:.3f} m"
)


print(
    f"RMSE GNSS disponible : "
    f"{gnss_available_rmse:.3f} m"
)


print(
    f"NEES moyen global : "
    f"{mean_nees:.3f}"
)


print(
    f"NEES théorique attendu : "
    f"6.000"
)


print()


print(
    "----- COUPURE GNSS -----"
)


print(
    f"RMSE pendant coupure : "
    f"{outage_rmse:.3f} m"
)


print(
    f"Erreur début coupure : "
    f"{error_at_outage_start:.3f} m"
)


print(
    f"Erreur fin coupure : "
    f"{error_at_outage_end:.3f} m"
)


print(
    f"Erreur maximale coupure : "
    f"{maximum_outage_error:.3f} m"
)


print()


print(
    f"Incertitude 3D début : "
    f"{sigma_at_outage_start:.3f} m"
)


print(
    f"Incertitude 3D fin : "
    f"{sigma_at_outage_end:.3f} m"
)


print()


print(
    f"NEES début coupure : "
    f"{nees_at_outage_start:.3f}"
)


print(
    f"NEES fin coupure : "
    f"{nees_at_outage_end:.3f}"
)


print()


print(
    "----- RETOUR GNSS -----"
)


print(
    f"Erreur première correction : "
    f"{first_return_error:.3f} m"
)


print(
    "============================================================"
)


# ------------------------------------------------------------
# 21. FIGURE POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    position_errors,
    label="Erreur réelle"
)


plt.plot(
    time_minutes,
    position_sigma_3d,
    linestyle="--",
    label="Incertitude 3D annoncée"
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
    "Position [m]"
)


plt.title(
    "AURORA — Effet d'une erreur de modèle J2"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. FIGURE NEES
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    nees_values,
    label="NEES"
)


plt.axhline(
    6.0,
    linestyle="--",
    label="Valeur moyenne théorique = 6"
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
    "AURORA — Cohérence de l'EKF avec erreur de modèle"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. FIGURE VITESSE
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    velocity_errors
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
    "Erreur vitesse [m/s]"
)


plt.title(
    "AURORA — Erreur de vitesse avec J2 non modélisé"
)


plt.grid(True)

plt.tight_layout()

plt.show()