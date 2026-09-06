import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

from src.dynamics.orbit import (
    R_EARTH,
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
# Expérience 007-B
# Validation statistique de l'EKF
#
# Outils :
#   - NIS  : Normalized Innovation Squared
#   - NEES : Normalized Estimation Error Squared
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE DE REFERENCE
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

simulation_duration = (
    60.0 * 60.0
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


# ------------------------------------------------------------
# 3. MONTE CARLO
# ------------------------------------------------------------

number_of_runs = 200

master_rng = np.random.default_rng(
    2026
)


# ------------------------------------------------------------
# 4. ETAT VRAI INITIAL
# ------------------------------------------------------------

true_position, true_velocity = (
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
        true_position,
        true_velocity
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
# 6. BRUIT DE MESURE GNSS
# ------------------------------------------------------------

gnss_position_sigma = 3.0

R = (
    gnss_position_sigma**2
    * np.eye(3)
)


# ------------------------------------------------------------
# 7. COVARIANCE INITIALE
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
# 8. BRUIT DE PROCESSUS
# ------------------------------------------------------------

# Ici nous voulons d'abord tester un cas
# "matched model" :
#
# vérité = dynamique deux corps
# filtre = dynamique deux corps
#
# On met donc Q = 0 pour cette validation
# mathématique initiale.

Q = np.zeros(
    (
        6,
        6
    )
)


# ------------------------------------------------------------
# 9. DIMENSIONS STATISTIQUES
# ------------------------------------------------------------

state_dimension = 6

measurement_dimension = 3

confidence_level = 0.95

alpha = (
    1.0 - confidence_level
)


# ------------------------------------------------------------
# 10. BORNES CHI2
# ------------------------------------------------------------

nis_lower_bound = chi2.ppf(
    alpha / 2.0,
    measurement_dimension
)

nis_upper_bound = chi2.ppf(
    1.0 - alpha / 2.0,
    measurement_dimension
)


nees_lower_bound = chi2.ppf(
    alpha / 2.0,
    state_dimension
)

nees_upper_bound = chi2.ppf(
    1.0 - alpha / 2.0,
    state_dimension
)


# ------------------------------------------------------------
# 11. STOCKAGE GLOBAL
# ------------------------------------------------------------

all_nis = []

all_nees = []

position_errors = []

position_sigma_values = []


# ------------------------------------------------------------
# 12. BOUCLE MONTE CARLO
# ------------------------------------------------------------

for run_index in range(
    number_of_runs
):

    # --------------------------------------------------------
    # Générateur indépendant pour ce run
    # --------------------------------------------------------

    run_seed = (
        master_rng.integers(
            0,
            2**32 - 1
        )
    )

    rng = np.random.default_rng(
        run_seed
    )


    # --------------------------------------------------------
    # Erreur initiale tirée selon P0
    # --------------------------------------------------------

    initial_error = (
        rng.multivariate_normal(
            mean=np.zeros(6),
            cov=P0
        )
    )


    estimated_state = (
        true_initial_state
        + initial_error
    )


    estimated_covariance = (
        P0.copy()
    )


    # --------------------------------------------------------
    # BOUCLE TEMPORELLE
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MESURE GNSS
        # ----------------------------------------------------

        measurement_noise = (
            rng.normal(
                loc=0.0,
                scale=gnss_position_sigma,
                size=3
            )
        )


        position_measurement = (
            truth_states[
                index,
                0:3
            ]
            +
            measurement_noise
        )


        # ----------------------------------------------------
        # MISE A JOUR EKF
        # ----------------------------------------------------

        update_result = (
            ekf_update_position(
                predicted_state=
                    predicted_state,

                predicted_covariance=
                    predicted_covariance,

                position_measurement=
                    position_measurement,

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


        # ----------------------------------------------------
        # NIS
        #
        # NIS = nu^T S^-1 nu
        # ----------------------------------------------------

        nis = (
            innovation.T
            @ np.linalg.solve(
                innovation_covariance,
                innovation
            )
        )


        all_nis.append(
            nis
        )


        # ----------------------------------------------------
        # ERREUR D'ESTIMATION
        # ----------------------------------------------------

        estimation_error = (
            estimated_state
            - truth_states[
                index
            ]
        )


        # ----------------------------------------------------
        # NEES
        #
        # NEES = e^T P^-1 e
        # ----------------------------------------------------

        nees = (
            estimation_error.T
            @ np.linalg.solve(
                estimated_covariance,
                estimation_error
            )
        )


        all_nees.append(
            nees
        )


        # ----------------------------------------------------
        # ERREUR POSITION
        # ----------------------------------------------------

        position_error = np.linalg.norm(
            estimation_error[
                0:3
            ]
        )


        position_errors.append(
            position_error
        )


        # ----------------------------------------------------
        # INCERTITUDE 3D
        # ----------------------------------------------------

        position_sigma = np.sqrt(
            np.trace(
                estimated_covariance[
                    0:3,
                    0:3
                ]
            )
        )


        position_sigma_values.append(
            position_sigma
        )


# ------------------------------------------------------------
# 13. CONVERSION NUMPY
# ------------------------------------------------------------

all_nis = np.array(
    all_nis
)

all_nees = np.array(
    all_nees
)

position_errors = np.array(
    position_errors
)

position_sigma_values = np.array(
    position_sigma_values
)


# ------------------------------------------------------------
# 14. STATISTIQUES NIS
# ------------------------------------------------------------

mean_nis = np.mean(
    all_nis
)

median_nis = np.median(
    all_nis
)


nis_inside_bounds = (
    (
        all_nis
        >= nis_lower_bound
    )
    &
    (
        all_nis
        <= nis_upper_bound
    )
)


nis_inside_percentage = (
    100.0
    * np.mean(
        nis_inside_bounds
    )
)


# ------------------------------------------------------------
# 15. STATISTIQUES NEES
# ------------------------------------------------------------

mean_nees = np.mean(
    all_nees
)

median_nees = np.median(
    all_nees
)


nees_inside_bounds = (
    (
        all_nees
        >= nees_lower_bound
    )
    &
    (
        all_nees
        <= nees_upper_bound
    )
)


nees_inside_percentage = (
    100.0
    * np.mean(
        nees_inside_bounds
    )
)


# ------------------------------------------------------------
# 16. STATISTIQUES POSITION
# ------------------------------------------------------------

position_rmse = np.sqrt(
    np.mean(
        position_errors**2
    )
)


mean_position_sigma = np.mean(
    position_sigma_values
)


# ------------------------------------------------------------
# 17. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================"
)

print(
    "AURORA — Expérience 007-B"
)

print(
    "Validation statistique NIS / NEES de l'EKF"
)

print(
    "============================================================"
)

print(
    f"Monte Carlo : "
    f"{number_of_runs} runs"
)

print(
    f"Durée par run : "
    f"{simulation_duration / 60.0:.1f} min"
)

print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)

print()

print(
    "----- NIS -----"
)

print(
    f"Dimension mesure : "
    f"{measurement_dimension}"
)

print(
    f"Valeur moyenne théorique : "
    f"{measurement_dimension:.3f}"
)

print(
    f"NIS moyen observé : "
    f"{mean_nis:.3f}"
)

print(
    f"NIS médian : "
    f"{median_nis:.3f}"
)

print(
    f"Intervalle Chi2 95 % : "
    f"[{nis_lower_bound:.3f}, "
    f"{nis_upper_bound:.3f}]"
)

print(
    f"Echantillons dans l'intervalle : "
    f"{nis_inside_percentage:.2f} %"
)

print()

print(
    "----- NEES -----"
)

print(
    f"Dimension état : "
    f"{state_dimension}"
)

print(
    f"Valeur moyenne théorique : "
    f"{state_dimension:.3f}"
)

print(
    f"NEES moyen observé : "
    f"{mean_nees:.3f}"
)

print(
    f"NEES médian : "
    f"{median_nees:.3f}"
)

print(
    f"Intervalle Chi2 95 % : "
    f"[{nees_lower_bound:.3f}, "
    f"{nees_upper_bound:.3f}]"
)

print(
    f"Echantillons dans l'intervalle : "
    f"{nees_inside_percentage:.2f} %"
)

print()

print(
    "----- POSITION -----"
)

print(
    f"RMSE position : "
    f"{position_rmse:.3f} m"
)

print(
    f"Incertitude position 3D moyenne : "
    f"{mean_position_sigma:.3f} m"
)

print(
    "============================================================"
)


# ------------------------------------------------------------
# 18. HISTOGRAMME NIS
# ------------------------------------------------------------

x_nis = np.linspace(
    0.0,
    np.percentile(
        all_nis,
        99.5
    ),
    400
)


plt.figure(
    figsize=(10, 5)
)


plt.hist(
    all_nis,
    bins=60,
    density=True,
    alpha=0.6,
    label="NIS observé"
)


plt.plot(
    x_nis,
    chi2.pdf(
        x_nis,
        measurement_dimension
    ),
    label="Chi2 théorique"
)


plt.axvline(
    nis_lower_bound,
    linestyle="--"
)


plt.axvline(
    nis_upper_bound,
    linestyle="--"
)


plt.xlabel(
    "NIS [-]"
)

plt.ylabel(
    "Densité"
)

plt.title(
    "AURORA — Distribution du NIS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. HISTOGRAMME NEES
# ------------------------------------------------------------

x_nees = np.linspace(
    0.0,
    np.percentile(
        all_nees,
        99.5
    ),
    400
)


plt.figure(
    figsize=(10, 5)
)


plt.hist(
    all_nees,
    bins=60,
    density=True,
    alpha=0.6,
    label="NEES observé"
)


plt.plot(
    x_nees,
    chi2.pdf(
        x_nees,
        state_dimension
    ),
    label="Chi2 théorique"
)


plt.axvline(
    nees_lower_bound,
    linestyle="--"
)


plt.axvline(
    nees_upper_bound,
    linestyle="--"
)


plt.xlabel(
    "NEES [-]"
)

plt.ylabel(
    "Densité"
)

plt.title(
    "AURORA — Distribution du NEES"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()