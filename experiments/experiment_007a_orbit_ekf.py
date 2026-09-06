import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH,
    orbital_period,
    propagate_orbit
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.navigation.orbit_ekf import (
    ekf_predict,
    ekf_update_position
)


# ============================================================
# AURORA
# Expérience 007-A
# EKF orbital avec coupure GNSS
#
# CORRECTION :
# La grille temporelle de l'EKF et celle de la
# vérité de référence sont exactement identiques.
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE VRAIE
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
# 2. PARAMETRES TEMPORELS
# ------------------------------------------------------------

period = orbital_period(
    semi_major_axis
)

simulation_duration = (
    2.0 * period
)

desired_dt = 10.0


# ------------------------------------------------------------
# IMPORTANT
#
# On choisit d'abord le nombre d'époques.
#
# Ensuite on génère UNE grille avec linspace.
#
# propagate_orbit() utilise lui aussi linspace,
# donc la vérité et l'EKF auront exactement
# les mêmes instants.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. COUPURE GNSS
# ------------------------------------------------------------

gnss_outage_start_minutes = 80.0

gnss_outage_end_minutes = 100.0


gnss_available = ~(
    (
        time / 60.0
        >= gnss_outage_start_minutes
    )
    &
    (
        time / 60.0
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
# 5. VERITE DE REFERENCE
# ------------------------------------------------------------

truth_solution = propagate_orbit(
    initial_state=
        true_initial_state,

    duration=
        simulation_duration,

    number_of_points=
        number_of_epochs
)


truth_states = (
    truth_solution.y.T
)


# ------------------------------------------------------------
# 6. VERIFICATION DE LA SYNCHRONISATION
# ------------------------------------------------------------

maximum_time_grid_error = np.max(
    np.abs(
        truth_solution.t
        - time
    )
)


# ------------------------------------------------------------
# 7. MESURES GNSS SIMPLIFIEES
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
# 9. COVARIANCE INITIALE
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
# 10. BRUIT DE PROCESSUS
# ------------------------------------------------------------

process_position_sigma = (
    0.02
)

process_velocity_sigma = (
    0.002
)


Q = np.diag([
    process_position_sigma**2,
    process_position_sigma**2,
    process_position_sigma**2,

    process_velocity_sigma**2,
    process_velocity_sigma**2,
    process_velocity_sigma**2
])


# ------------------------------------------------------------
# 11. BRUIT DES MESURES GNSS
# ------------------------------------------------------------

R = (
    gnss_position_sigma**2
    * np.eye(3)
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


innovation_norms = np.full(
    number_of_epochs,
    np.nan
)


estimated_states[0] = (
    estimated_state
)


position_errors[0] = np.linalg.norm(
    estimated_state[0:3]
    - truth_states[0, 0:3]
)


velocity_errors[0] = np.linalg.norm(
    estimated_state[3:6]
    - truth_states[0, 3:6]
)


position_sigma_3d[0] = np.sqrt(
    np.trace(
        estimated_covariance[
            0:3,
            0:3
        ]
    )
)


# ------------------------------------------------------------
# 13. BOUCLE EKF
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # --------------------------------------------------------
    # PREDICTION
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

        # ----------------------------------------------------
        # GNSS indisponible :
        # aucune correction
        # ----------------------------------------------------

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


    position_errors[
        index
    ] = np.linalg.norm(
        estimated_state[
            0:3
        ]
        - truth_states[
            index,
            0:3
        ]
    )


    velocity_errors[
        index
    ] = np.linalg.norm(
        estimated_state[
            3:6
        ]
        - truth_states[
            index,
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


# ------------------------------------------------------------
# 14. MASQUES
# ------------------------------------------------------------

outage_mask = (
    ~gnss_available
)

available_mask = (
    gnss_available
)


# ------------------------------------------------------------
# 15. STATISTIQUES
# ------------------------------------------------------------

rmse_position_all = np.sqrt(
    np.mean(
        position_errors**2
    )
)


rmse_position_gnss = np.sqrt(
    np.mean(
        position_errors[
            available_mask
        ]**2
    )
)


rmse_position_outage = np.sqrt(
    np.mean(
        position_errors[
            outage_mask
        ]**2
    )
)


maximum_position_error_outage = np.max(
    position_errors[
        outage_mask
    ]
)


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


# ------------------------------------------------------------
# 16. RETOUR DU GNSS
# ------------------------------------------------------------

after_outage_indices = np.where(
    (
        time / 60.0
        > gnss_outage_end_minutes
    )
)[0]


if len(
    after_outage_indices
) > 0:

    first_after_outage = (
        after_outage_indices[0]
    )


    error_first_after_outage = (
        position_errors[
            first_after_outage
        ]
    )

else:

    error_first_after_outage = (
        np.nan
    )


# ------------------------------------------------------------
# 17. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================"
)

print(
    "AURORA — Expérience 007-A CORRIGEE"
)

print(
    "EKF orbital et coupure GNSS"
)

print(
    "============================================================"
)


print(
    f"Pas temporel demandé : "
    f"{desired_dt:.6f} s"
)


print(
    f"Pas temporel réellement utilisé : "
    f"{dt:.9f} s"
)


print(
    f"Erreur maximale de synchronisation temporelle : "
    f"{maximum_time_grid_error:.6e} s"
)


print(
    f"Durée simulée : "
    f"{simulation_duration / 60.0:.2f} min"
)


print(
    f"Coupure GNSS : "
    f"{gnss_outage_start_minutes:.1f} "
    f"à {gnss_outage_end_minutes:.1f} min"
)


print()


print(
    f"Erreur position initiale : "
    f"{position_errors[0]:.3f} m"
)


print()


print(
    "----- PERFORMANCE GLOBALE -----"
)


print(
    f"RMSE position global : "
    f"{rmse_position_all:.3f} m"
)


print(
    f"RMSE avec GNSS disponible : "
    f"{rmse_position_gnss:.3f} m"
)


print()


print(
    "----- PENDANT LA COUPURE -----"
)


print(
    f"RMSE position : "
    f"{rmse_position_outage:.3f} m"
)


print(
    f"Erreur au début : "
    f"{error_at_outage_start:.3f} m"
)


print(
    f"Erreur à la fin : "
    f"{error_at_outage_end:.3f} m"
)


print(
    f"Erreur maximale : "
    f"{maximum_position_error_outage:.3f} m"
)


print()


print(
    f"Incertitude 3D début coupure : "
    f"{sigma_at_outage_start:.3f} m"
)


print(
    f"Incertitude 3D fin coupure : "
    f"{sigma_at_outage_end:.3f} m"
)


print()


print(
    "----- RETOUR DU GNSS -----"
)


print(
    f"Erreur à la première correction "
    f"après coupure : "
    f"{error_first_after_outage:.3f} m"
)


print(
    "============================================================"
)


# ------------------------------------------------------------
# 18. FIGURE ERREUR POSITION
# ------------------------------------------------------------

time_minutes = (
    time / 60.0
)


plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    position_errors,
    label="Erreur EKF"
)


plt.plot(
    time_minutes,
    position_sigma_3d,
    linestyle="--",
    label="Incertitude position 3D"
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
    "AURORA — EKF orbital pendant une coupure GNSS"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FIGURE ERREUR VITESSE
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
    "Erreur de vitesse [m/s]"
)


plt.title(
    "AURORA — Erreur de vitesse de l'EKF"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. FIGURE INNOVATIONS
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    innovation_norms
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
    "Norme innovation GNSS [m]"
)


plt.title(
    "AURORA — Innovations de l'EKF"
)


plt.grid(True)

plt.tight_layout()

plt.show()