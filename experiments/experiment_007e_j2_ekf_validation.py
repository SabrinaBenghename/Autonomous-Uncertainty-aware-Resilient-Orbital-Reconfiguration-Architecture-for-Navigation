import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.perturbations import (
    propagate_orbit_with_j2
)

from src.navigation.orbit_ekf import (
    ekf_update_position
)

from src.navigation.orbit_ekf_models import (
    j2_state_derivative,
    ekf_predict_custom,
    ekf_predict_variational
)


# ============================================================
# AURORA
# Expérience 007-E
#
# Validation Monte Carlo de l'EKF J2
#
# Comparaison :
#
# 1. STM premier ordre :
#       Phi ≈ I + F dt
#
# 2. STM variationnelle :
#       Phi_dot = F Phi
#
# Vérité :
#       Two-body + J2
#
# Filtre :
#       Two-body + J2
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


time_minutes = (
    time / 60.0
)


# ------------------------------------------------------------
# 3. COUPURE GNSS
# ------------------------------------------------------------

gnss_outage_start_minutes = (
    20.0
)

gnss_outage_end_minutes = (
    40.0
)


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


outage_mask = (
    ~gnss_available
)


outage_indices = np.where(
    outage_mask
)[0]


outage_end_index = (
    outage_indices[-1]
)


# ------------------------------------------------------------
# 4. MONTE CARLO
# ------------------------------------------------------------

number_of_runs = 50


master_rng = np.random.default_rng(
    2026
)


run_seeds = (
    master_rng.integers(
        low=0,
        high=2**32 - 1,
        size=number_of_runs
    )
)


# ------------------------------------------------------------
# 5. ETAT INITIAL VRAI
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
# 6. VERITE J2
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


maximum_time_error = np.max(
    np.abs(
        truth_solution.t
        - time
    )
)


# ------------------------------------------------------------
# 7. BRUIT GNSS
# ------------------------------------------------------------

gnss_position_sigma = (
    3.0
)


R = (
    gnss_position_sigma**2
    * np.eye(3)
)


# ------------------------------------------------------------
# 8. COVARIANCE INITIALE
# ------------------------------------------------------------

initial_position_sigma = (
    100.0
)

initial_velocity_sigma = (
    0.10
)


P0 = np.diag([
    initial_position_sigma**2,
    initial_position_sigma**2,
    initial_position_sigma**2,

    initial_velocity_sigma**2,
    initial_velocity_sigma**2,
    initial_velocity_sigma**2
])


# ------------------------------------------------------------
# 9. PROCESS NOISE
#
# Modèle parfaitement apparié :
#
# vérité = J2
# filtre = J2
#
# Donc Q = 0 pour cette validation.
# ------------------------------------------------------------

Q = np.zeros(
    (
        6,
        6
    )
)


# ------------------------------------------------------------
# 10. BORNES CHI2
# ------------------------------------------------------------

confidence_level = (
    0.95
)

alpha = (
    1.0
    - confidence_level
)


measurement_dimension = (
    3
)

state_dimension = (
    6
)


nis_lower = chi2.ppf(
    alpha / 2.0,
    measurement_dimension
)


nis_upper = chi2.ppf(
    1.0 - alpha / 2.0,
    measurement_dimension
)


nees_lower = chi2.ppf(
    alpha / 2.0,
    state_dimension
)


nees_upper = chi2.ppf(
    1.0 - alpha / 2.0,
    state_dimension
)


# ------------------------------------------------------------
# 11. FONCTION MONTE CARLO
# ------------------------------------------------------------

def evaluate_predictor(
    predictor_name,
    predictor_function
):

    all_nis = []

    all_nees = []

    all_nees_outage = []

    all_position_errors = []

    all_position_errors_outage = []

    outage_end_errors = []

    outage_end_sigmas = []


    # --------------------------------------------------------
    # Boucle Monte Carlo
    # --------------------------------------------------------

    for run_index in range(
        number_of_runs
    ):

        rng = np.random.default_rng(
            run_seeds[
                run_index
            ]
        )


        # ----------------------------------------------------
        # Erreur initiale cohérente avec P0
        # ----------------------------------------------------

        initial_error = (
            rng.multivariate_normal(
                mean=
                    np.zeros(6),

                cov=
                    P0
            )
        )


        estimated_state = (
            true_initial_state
            + initial_error
        )


        covariance = (
            P0.copy()
        )


        # ----------------------------------------------------
        # Mesures communes à ce run
        # ----------------------------------------------------

        measurement_noise = (
            rng.normal(
                loc=0.0,
                scale=gnss_position_sigma,
                size=(
                    number_of_epochs,
                    3
                )
            )
        )


        gnss_measurements = (
            truth_states[:, 0:3]
            + measurement_noise
        )


        # ----------------------------------------------------
        # Boucle temporelle
        # ----------------------------------------------------

        for index in range(
            1,
            number_of_epochs
        ):

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            (
                predicted_state,
                predicted_covariance
            ) = predictor_function(
                state=
                    estimated_state,

                covariance=
                    covariance,

                dt=
                    dt,

                process_noise_covariance=
                    Q,

                dynamics_function=
                    j2_state_derivative
            )


            # ------------------------------------------------
            # GNSS disponible
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Erreur vraie
            # ------------------------------------------------

            estimation_error = (
                estimated_state
                - truth_states[
                    index
                ]
            )


            position_error = np.linalg.norm(
                estimation_error[
                    0:3
                ]
            )


            all_position_errors.append(
                position_error
            )


            # ------------------------------------------------
            # NEES
            # ------------------------------------------------

            try:

                nees = (
                    estimation_error.T
                    @ np.linalg.solve(
                        covariance,
                        estimation_error
                    )
                )

            except np.linalg.LinAlgError:

                nees = (
                    np.nan
                )


            if np.isfinite(
                nees
            ):

                all_nees.append(
                    nees
                )


                if outage_mask[
                    index
                ]:

                    all_nees_outage.append(
                        nees
                    )


            # ------------------------------------------------
            # Erreur pendant outage
            # ------------------------------------------------

            if outage_mask[
                index
            ]:

                all_position_errors_outage.append(
                    position_error
                )


            # ------------------------------------------------
            # Fin de coupure
            # ------------------------------------------------

            if index == outage_end_index:

                outage_end_errors.append(
                    position_error
                )


                position_sigma_3d = np.sqrt(
                    np.trace(
                        covariance[
                            0:3,
                            0:3
                        ]
                    )
                )


                outage_end_sigmas.append(
                    position_sigma_3d
                )


    # --------------------------------------------------------
    # Conversion
    # --------------------------------------------------------

    all_nis = np.array(
        all_nis
    )

    all_nees = np.array(
        all_nees
    )

    all_nees_outage = np.array(
        all_nees_outage
    )

    all_position_errors = np.array(
        all_position_errors
    )

    all_position_errors_outage = np.array(
        all_position_errors_outage
    )

    outage_end_errors = np.array(
        outage_end_errors
    )

    outage_end_sigmas = np.array(
        outage_end_sigmas
    )


    # --------------------------------------------------------
    # Statistiques
    # --------------------------------------------------------

    mean_nis = np.mean(
        all_nis
    )


    nis_coverage = (
        100.0
        * np.mean(
            (
                all_nis >= nis_lower
            )
            &
            (
                all_nis <= nis_upper
            )
        )
    )


    mean_nees = np.mean(
        all_nees
    )


    nees_coverage = (
        100.0
        * np.mean(
            (
                all_nees >= nees_lower
            )
            &
            (
                all_nees <= nees_upper
            )
        )
    )


    mean_outage_nees = np.mean(
        all_nees_outage
    )


    position_rmse = np.sqrt(
        np.mean(
            all_position_errors**2
        )
    )


    outage_position_rmse = np.sqrt(
        np.mean(
            all_position_errors_outage**2
        )
    )


    outage_end_rmse = np.sqrt(
        np.mean(
            outage_end_errors**2
        )
    )


    mean_outage_end_sigma = np.mean(
        outage_end_sigmas
    )


    return {
        "name":
            predictor_name,

        "mean_nis":
            mean_nis,

        "nis_coverage":
            nis_coverage,

        "mean_nees":
            mean_nees,

        "nees_coverage":
            nees_coverage,

        "mean_outage_nees":
            mean_outage_nees,

        "position_rmse":
            position_rmse,

        "outage_position_rmse":
            outage_position_rmse,

        "outage_end_rmse":
            outage_end_rmse,

        "outage_end_sigma":
            mean_outage_end_sigma,

        "all_nis":
            all_nis,

        "all_nees":
            all_nees
    }


# ------------------------------------------------------------
# 12. EVALUATION ANCIENNE STM
# ------------------------------------------------------------

result_first_order = (
    evaluate_predictor(
        predictor_name=
            "Phi = I + F dt",

        predictor_function=
            ekf_predict_custom
    )
)


# ------------------------------------------------------------
# 13. EVALUATION STM VARIATIONNELLE
# ------------------------------------------------------------

result_variational = (
    evaluate_predictor(
        predictor_name=
            "STM variationnelle",

        predictor_function=
            ekf_predict_variational
    )
)


results = [
    result_first_order,
    result_variational
]


# ------------------------------------------------------------
# 14. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)

print(
    "AURORA — Expérience 007-E"
)

print(
    "Validation Monte Carlo de la covariance EKF J2"
)

print(
    "============================================================================================"
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
    f"Coupure GNSS : "
    f"{gnss_outage_start_minutes:.1f} "
    f"à {gnss_outage_end_minutes:.1f} min"
)


print(
    f"Pas temporel : "
    f"{dt:.6f} s"
)


print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)


print()


print(
    "Références : "
    "NIS moyen ≈ 3 ; "
    "NEES moyen ≈ 6 ; "
    "couverture ≈ 95 %"
)


print()


print(
    "Méthode            | NIS moy | NIS 95% | "
    "NEES moy | NEES 95% | NEES outage | "
    "RMSE pos | RMSE outage | Fin outage | Sigma fin"
)


print(
    "                   |         |   [%]   | "
    "         |   [%]    |             | "
    "  [m]    |    [m]      |    [m]     |   [m]"
)


print(
    "--------------------------------------------------------------------------------------------"
)


for result in results:

    print(
        f"{result['name']:<18} | "
        f"{result['mean_nis']:>7.3f} | "
        f"{result['nis_coverage']:>7.2f} | "
        f"{result['mean_nees']:>8.3f} | "
        f"{result['nees_coverage']:>8.2f} | "
        f"{result['mean_outage_nees']:>11.3f} | "
        f"{result['position_rmse']:>8.3f} | "
        f"{result['outage_position_rmse']:>11.3f} | "
        f"{result['outage_end_rmse']:>10.3f} | "
        f"{result['outage_end_sigma']:>9.3f}"
    )


print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 15. HISTOGRAMME NIS
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.hist(
    result_variational[
        "all_nis"
    ],
    bins=60,
    density=True,
    alpha=0.6,
    label="NIS STM variationnelle"
)


x_nis = np.linspace(
    0.0,
    np.percentile(
        result_variational[
            "all_nis"
        ],
        99.5
    ),
    400
)


plt.plot(
    x_nis,
    chi2.pdf(
        x_nis,
        measurement_dimension
    ),
    label="Chi2 théorique"
)


plt.xlabel(
    "NIS [-]"
)


plt.ylabel(
    "Densité"
)


plt.title(
    "AURORA — NIS avec STM variationnelle"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. HISTOGRAMME NEES
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)


plt.hist(
    result_variational[
        "all_nees"
    ],
    bins=60,
    density=True,
    alpha=0.6,
    label="NEES STM variationnelle"
)


x_nees = np.linspace(
    0.0,
    np.percentile(
        result_variational[
            "all_nees"
        ],
        99.5
    ),
    400
)


plt.plot(
    x_nees,
    chi2.pdf(
        x_nees,
        state_dimension
    ),
    label="Chi2 théorique"
)


plt.xlabel(
    "NEES [-]"
)


plt.ylabel(
    "Densité"
)


plt.title(
    "AURORA — NEES avec STM variationnelle"
)


plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()