import numpy as np

from src.dynamics.orbit import (
    R_EARTH
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.dynamics.non_gravitational_forces import (
    synthetic_drag_like_acceleration,
    propagate_orbit_with_j2_and_drag_like
)

from src.sensors.accelerometer import (
    simulate_accelerometer_measurement
)

from src.navigation.accelerometer_bias_ekf import (
    ekf_predict_with_bias_estimation,
    ekf_update_position_augmented
)

from src.navigation.gnss_ekf_interface import (
    generate_simplified_gps_positions,
    select_visible_gnss_satellites,
    simulate_pseudorange_measurements,
    solve_gnss_least_squares,
    compute_gnss_solution_covariance
)

from src.navigation.gnss_fdir_interface import (
    solve_gnss_with_fdir
)


# ============================================================
# AURORA
# Experience 009-C
#
# Monte Carlo du systeme integre :
#
# GNSS pseudoranges
#       ↓
# faute GNSS
#       ↓
# FDIR
#       ↓
# EKF 9D + accelerometre
#
# Comparaison :
#
# - EKF non protege
# - EKF protege FDIR
# ============================================================


# ------------------------------------------------------------
# 1. CONFIGURATION MONTE CARLO
# ------------------------------------------------------------

number_of_runs = (
    30
)


master_rng = np.random.default_rng(
    2026
)


run_seeds = master_rng.integers(
    low=0,
    high=2**32 - 1,
    size=number_of_runs
)


# ------------------------------------------------------------
# 2. ORBITE
# ------------------------------------------------------------

semi_major_axis = (
    R_EARTH
    + 550_000.0
)

eccentricity = (
    0.01
)

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
# 3. TEMPS
# ------------------------------------------------------------

simulation_duration_minutes = (
    130.0
)

simulation_duration = (
    simulation_duration_minutes
    * 60.0
)

desired_dt = (
    10.0
)


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
    time
    / 60.0
)


# ------------------------------------------------------------
# 4. FAUTE GNSS
# ------------------------------------------------------------

fault_start_minutes = (
    60.0
)

fault_end_minutes = (
    80.0
)

fault_bias_meters = (
    50.0
)


fault_window_mask = (
    (
        time_minutes
        >= fault_start_minutes
    )
    &
    (
        time_minutes
        <= fault_end_minutes
    )
)


# ------------------------------------------------------------
# 5. COUPURE GNSS
# ------------------------------------------------------------

outage_start_minutes = (
    90.0
)

outage_end_minutes = (
    110.0
)


forced_gnss_available = ~(
    (
        time_minutes
        >= outage_start_minutes
    )
    &
    (
        time_minutes
        <= outage_end_minutes
    )
)


outage_mask = (
    ~forced_gnss_available
)


# ------------------------------------------------------------
# 6. VERITE ORBITALE
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


true_initial_orbital_state = np.concatenate(
    (
        true_initial_position,
        true_initial_velocity
    )
)


true_non_gravitational_acceleration = (
    2.0e-5
)


truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_orbital_state,

        duration=
            simulation_duration,

        number_of_points=
            number_of_epochs,

        base_acceleration=
            true_non_gravitational_acceleration
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
# 7. FORCE SPECIFIQUE VRAIE
# ------------------------------------------------------------

true_specific_forces = np.zeros(
    (
        number_of_epochs,
        3
    )
)


for index in range(
    number_of_epochs
):

    true_specific_forces[
        index
    ] = (
        synthetic_drag_like_acceleration(
            time=
                time[index],

            position=
                truth_states[
                    index,
                    0:3
                ],

            velocity=
                truth_states[
                    index,
                    3:6
                ],

            base_acceleration=
                true_non_gravitational_acceleration
        )
    )


# ------------------------------------------------------------
# 8. GNSS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


fdir_confidence = (
    0.99
)


# ------------------------------------------------------------
# 9. ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


initial_bias_sigma = (
    5.0e-6
)


bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 10. INCERTITUDE INITIALE
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
    initial_velocity_sigma**2,

    initial_bias_sigma**2,
    initial_bias_sigma**2,
    initial_bias_sigma**2
])


# ------------------------------------------------------------
# 11. CHOIX DU SATELLITE FAUTIF
# ------------------------------------------------------------

visibility_counter = {
    f"GPS{index:02d}": 0
    for index in range(
        1,
        25
    )
}


fault_indices = np.where(
    fault_window_mask
)[0]


for index in fault_indices:

    (
        all_positions,
        all_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[index]
    )


    (
        _,
        visible_ids
    ) = select_visible_gnss_satellites(
        receiver_position=
            truth_states[
                index,
                0:3
            ],

        satellite_positions=
            all_positions,

        satellite_ids=
            all_ids
    )


    for satellite_id in visible_ids:

        visibility_counter[
            satellite_id
        ] += 1


faulty_satellite_id = max(
    visibility_counter,
    key=visibility_counter.get
)


# ------------------------------------------------------------
# 12. STOCKAGE GLOBAL
# ------------------------------------------------------------

unprotected_fault_rmse_runs = np.zeros(
    number_of_runs
)

protected_fault_rmse_runs = np.zeros(
    number_of_runs
)


unprotected_outage_rmse_runs = np.zeros(
    number_of_runs
)

protected_outage_rmse_runs = np.zeros(
    number_of_runs
)


unprotected_end_outage_errors = np.zeros(
    number_of_runs
)

protected_end_outage_errors = np.zeros(
    number_of_runs
)


unprotected_end_outage_sigmas = np.zeros(
    number_of_runs
)

protected_end_outage_sigmas = np.zeros(
    number_of_runs
)


unprotected_nis_means = np.zeros(
    number_of_runs
)

protected_nis_means = np.zeros(
    number_of_runs
)


detection_rates = np.zeros(
    number_of_runs
)

isolation_rates = np.zeros(
    number_of_runs
)

false_alarm_rates = np.zeros(
    number_of_runs
)


# ------------------------------------------------------------
# 13. INDICE FIN OUTAGE
# ------------------------------------------------------------

outage_indices = np.where(
    outage_mask
)[0]


outage_end_index = (
    outage_indices[-1]
)


# ------------------------------------------------------------
# 14. BOUCLE MONTE CARLO
# ------------------------------------------------------------

for run_index in range(
    number_of_runs
):

    rng = np.random.default_rng(
        run_seeds[
            run_index
        ]
    )


    # --------------------------------------------------------
    # Biais accelerometre vrai
    # --------------------------------------------------------

    true_accelerometer_bias = rng.normal(
        loc=0.0,
        scale=initial_bias_sigma,
        size=3
    )


    # --------------------------------------------------------
    # Mesures accelerometre
    # --------------------------------------------------------

    accelerometer_measurements = np.zeros(
        (
            number_of_epochs,
            3
        )
    )


    for index in range(
        number_of_epochs
    ):

        measurement_result = (
            simulate_accelerometer_measurement(
                true_specific_force_eci=
                    true_specific_forces[
                        index
                    ],

                bias_eci=
                    true_accelerometer_bias,

                noise_std=
                    accelerometer_noise_std,

                rng=
                    rng
            )
        )


        accelerometer_measurements[
            index
        ] = (
            measurement_result[
                "measurement"
            ]
        )


    # --------------------------------------------------------
    # Erreurs initiales
    # --------------------------------------------------------

    initial_position_error = rng.normal(
        loc=0.0,
        scale=initial_position_sigma,
        size=3
    )


    initial_velocity_error = rng.normal(
        loc=0.0,
        scale=initial_velocity_sigma,
        size=3
    )


    initial_augmented_state = np.concatenate(
        (
            true_initial_orbital_state
            +
            np.concatenate(
                (
                    initial_position_error,
                    initial_velocity_error
                )
            ),

            np.zeros(3)
        )
    )


    unprotected_state = (
        initial_augmented_state.copy()
    )

    protected_state = (
        initial_augmented_state.copy()
    )


    unprotected_covariance = (
        P0.copy()
    )

    protected_covariance = (
        P0.copy()
    )


    unprotected_clock_bias_meters = (
        0.0
    )

    protected_clock_bias_meters = (
        0.0
    )


    # --------------------------------------------------------
    # Stockage run
    # --------------------------------------------------------

    unprotected_position_errors = np.zeros(
        number_of_epochs
    )

    protected_position_errors = np.zeros(
        number_of_epochs
    )


    unprotected_position_sigmas = np.zeros(
        number_of_epochs
    )

    protected_position_sigmas = np.zeros(
        number_of_epochs
    )


    unprotected_nis = np.full(
        number_of_epochs,
        np.nan
    )

    protected_nis = np.full(
        number_of_epochs,
        np.nan
    )


    fault_active = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    fault_detected = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    correct_isolation = np.zeros(
        number_of_epochs,
        dtype=bool
    )

    fdir_attempted = np.zeros(
        number_of_epochs,
        dtype=bool
    )


    # --------------------------------------------------------
    # Etat initial
    # --------------------------------------------------------

    true_initial_augmented_state = np.concatenate(
        (
            truth_states[0],
            true_accelerometer_bias
        )
    )


    unprotected_error = (
        unprotected_state
        - true_initial_augmented_state
    )

    protected_error = (
        protected_state
        - true_initial_augmented_state
    )


    unprotected_position_errors[
        0
    ] = np.linalg.norm(
        unprotected_error[
            0:3
        ]
    )


    protected_position_errors[
        0
    ] = np.linalg.norm(
        protected_error[
            0:3
        ]
    )


    unprotected_position_sigmas[
        0
    ] = np.sqrt(
        np.trace(
            unprotected_covariance[
                0:3,
                0:3
            ]
        )
    )


    protected_position_sigmas[
        0
    ] = np.sqrt(
        np.trace(
            protected_covariance[
                0:3,
                0:3
            ]
        )
    )


    # --------------------------------------------------------
    # BOUCLE TEMPORELLE
    # --------------------------------------------------------

    for index in range(
        1,
        number_of_epochs
    ):

        # ----------------------------------------------------
        # Prediction EKF
        # ----------------------------------------------------

        (
            predicted_unprotected_state,
            predicted_unprotected_covariance
        ) = ekf_predict_with_bias_estimation(
            state=
                unprotected_state,

            covariance=
                unprotected_covariance,

            dt=
                dt,

            accelerometer_measurement_eci=
                accelerometer_measurements[
                    index - 1
                ],

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                bias_random_walk_density
        )


        (
            predicted_protected_state,
            predicted_protected_covariance
        ) = ekf_predict_with_bias_estimation(
            state=
                protected_state,

            covariance=
                protected_covariance,

            dt=
                dt,

            accelerometer_measurement_eci=
                accelerometer_measurements[
                    index - 1
                ],

            accelerometer_noise_std=
                accelerometer_noise_std,

            bias_random_walk_density=
                bias_random_walk_density
        )


        # ----------------------------------------------------
        # GNSS constellation
        # ----------------------------------------------------

        (
            all_positions,
            all_ids
        ) = generate_simplified_gps_positions(
            time_seconds=
                time[index]
        )


        (
            visible_positions,
            visible_ids
        ) = select_visible_gnss_satellites(
            receiver_position=
                truth_states[
                    index,
                    0:3
                ],

            satellite_positions=
                all_positions,

            satellite_ids=
                all_ids
        )


        can_use_gnss = (
            forced_gnss_available[
                index
            ]
            and
            len(
                visible_positions
            ) >= 4
        )


        if can_use_gnss:

            # ------------------------------------------------
            # Pseudoranges communs
            # ------------------------------------------------

            pseudorange_result = (
                simulate_pseudorange_measurements(
                    receiver_position=
                        truth_states[
                            index,
                            0:3
                        ],

                    satellite_positions=
                        visible_positions,

                    receiver_clock_bias_seconds=
                        receiver_clock_bias_seconds,

                    pseudorange_noise_std=
                        pseudorange_noise_std,

                    rng=
                        rng
                )
            )


            pseudoranges = (
                pseudorange_result[
                    "pseudoranges"
                ].copy()
            )


            # ------------------------------------------------
            # Injection de faute
            # ------------------------------------------------

            if (
                fault_window_mask[
                    index
                ]
                and
                faulty_satellite_id
                in visible_ids
            ):

                faulty_index = (
                    visible_ids.index(
                        faulty_satellite_id
                    )
                )


                pseudoranges[
                    faulty_index
                ] += (
                    fault_bias_meters
                )


                fault_active[
                    index
                ] = (
                    True
                )


            # =================================================
            # FILTRE NON PROTEGE
            # =================================================

            try:

                gnss_result = (
                    solve_gnss_least_squares(
                        satellite_positions=
                            visible_positions,

                        pseudoranges=
                            pseudoranges,

                        initial_position=
                            predicted_unprotected_state[
                                0:3
                            ],

                        initial_clock_bias_meters=
                            unprotected_clock_bias_meters
                    )
                )


                if gnss_result[
                    "converged"
                ]:

                    gnss_position = (
                        gnss_result[
                            "position"
                        ]
                    )


                    unprotected_clock_bias_meters = (
                        gnss_result[
                            "clock_bias_meters"
                        ]
                    )


                    covariance_result = (
                        compute_gnss_solution_covariance(
                            receiver_position=
                                gnss_position,

                            satellite_positions=
                                visible_positions,

                            pseudorange_noise_std=
                                pseudorange_noise_std
                        )
                    )


                    R_gnss = (
                        covariance_result[
                            "position_covariance"
                        ]
                    )


                    update_result = (
                        ekf_update_position_augmented(
                            predicted_state=
                                predicted_unprotected_state,

                            predicted_covariance=
                                predicted_unprotected_covariance,

                            position_measurement=
                                gnss_position,

                            measurement_noise_covariance=
                                R_gnss
                        )
                    )


                    innovation = (
                        update_result[
                            "innovation"
                        ]
                    )


                    S = (
                        update_result[
                            "innovation_covariance"
                        ]
                    )


                    unprotected_nis[
                        index
                    ] = (
                        innovation.T
                        @ np.linalg.solve(
                            S,
                            innovation
                        )
                    )


                    unprotected_state = (
                        update_result[
                            "state"
                        ]
                    )


                    unprotected_covariance = (
                        update_result[
                            "covariance"
                        ]
                    )


                else:

                    unprotected_state = (
                        predicted_unprotected_state
                    )

                    unprotected_covariance = (
                        predicted_unprotected_covariance
                    )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError
            ):

                unprotected_state = (
                    predicted_unprotected_state
                )

                unprotected_covariance = (
                    predicted_unprotected_covariance
                )


            # =================================================
            # FILTRE PROTEGE
            # =================================================

            try:

                fdir_result = (
                    solve_gnss_with_fdir(
                        satellite_positions=
                            visible_positions,

                        pseudoranges=
                            pseudoranges,

                        satellite_ids=
                            visible_ids,

                        initial_position=
                            predicted_protected_state[
                                0:3
                            ],

                        initial_clock_bias_meters=
                            protected_clock_bias_meters,

                        pseudorange_noise_std=
                            pseudorange_noise_std,

                        confidence=
                            fdir_confidence
                    )
                )


                fdir_attempted[
                    index
                ] = (
                    True
                )


                fault_detected[
                    index
                ] = (
                    fdir_result[
                        "fault_detected"
                    ]
                )


                isolated_id = (
                    fdir_result.get(
                        "isolated_id",
                        None
                    )
                )


                if (
                    fault_active[
                        index
                    ]
                    and
                    isolated_id
                    ==
                    faulty_satellite_id
                ):

                    correct_isolation[
                        index
                    ] = (
                        True
                    )


                if fdir_result[
                    "measurement_accepted"
                ]:

                    protected_solution = (
                        fdir_result[
                            "solution"
                        ]
                    )


                    protected_positions_used = (
                        fdir_result[
                            "satellite_positions"
                        ]
                    )


                    protected_position = (
                        protected_solution[
                            "position"
                        ]
                    )


                    protected_clock_bias_meters = (
                        protected_solution[
                            "clock_bias_meters"
                        ]
                    )


                    covariance_result = (
                        compute_gnss_solution_covariance(
                            receiver_position=
                                protected_position,

                            satellite_positions=
                                protected_positions_used,

                            pseudorange_noise_std=
                                pseudorange_noise_std
                        )
                    )


                    R_gnss = (
                        covariance_result[
                            "position_covariance"
                        ]
                    )


                    update_result = (
                        ekf_update_position_augmented(
                            predicted_state=
                                predicted_protected_state,

                            predicted_covariance=
                                predicted_protected_covariance,

                            position_measurement=
                                protected_position,

                            measurement_noise_covariance=
                                R_gnss
                        )
                    )


                    innovation = (
                        update_result[
                            "innovation"
                        ]
                    )


                    S = (
                        update_result[
                            "innovation_covariance"
                        ]
                    )


                    protected_nis[
                        index
                    ] = (
                        innovation.T
                        @ np.linalg.solve(
                            S,
                            innovation
                        )
                    )


                    protected_state = (
                        update_result[
                            "state"
                        ]
                    )


                    protected_covariance = (
                        update_result[
                            "covariance"
                        ]
                    )


                else:

                    protected_state = (
                        predicted_protected_state
                    )

                    protected_covariance = (
                        predicted_protected_covariance
                    )


            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError
            ):

                protected_state = (
                    predicted_protected_state
                )

                protected_covariance = (
                    predicted_protected_covariance
                )


        else:

            unprotected_state = (
                predicted_unprotected_state
            )

            unprotected_covariance = (
                predicted_unprotected_covariance
            )


            protected_state = (
                predicted_protected_state
            )

            protected_covariance = (
                predicted_protected_covariance
            )


        # ----------------------------------------------------
        # Erreurs
        # ----------------------------------------------------

        true_augmented_state = np.concatenate(
            (
                truth_states[
                    index
                ],

                true_accelerometer_bias
            )
        )


        unprotected_error = (
            unprotected_state
            - true_augmented_state
        )


        protected_error = (
            protected_state
            - true_augmented_state
        )


        unprotected_position_errors[
            index
        ] = np.linalg.norm(
            unprotected_error[
                0:3
            ]
        )


        protected_position_errors[
            index
        ] = np.linalg.norm(
            protected_error[
                0:3
            ]
        )


        unprotected_position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                unprotected_covariance[
                    0:3,
                    0:3
                ]
            )
        )


        protected_position_sigmas[
            index
        ] = np.sqrt(
            np.trace(
                protected_covariance[
                    0:3,
                    0:3
                ]
            )
        )


    # --------------------------------------------------------
    # STATISTIQUES DU RUN
    # --------------------------------------------------------

    actual_fault_mask = (
        fault_active
    )


    unprotected_fault_rmse_runs[
        run_index
    ] = np.sqrt(
        np.mean(
            unprotected_position_errors[
                actual_fault_mask
            ]**2
        )
    )


    protected_fault_rmse_runs[
        run_index
    ] = np.sqrt(
        np.mean(
            protected_position_errors[
                actual_fault_mask
            ]**2
        )
    )


    unprotected_outage_rmse_runs[
        run_index
    ] = np.sqrt(
        np.mean(
            unprotected_position_errors[
                outage_mask
            ]**2
        )
    )


    protected_outage_rmse_runs[
        run_index
    ] = np.sqrt(
        np.mean(
            protected_position_errors[
                outage_mask
            ]**2
        )
    )


    unprotected_end_outage_errors[
        run_index
    ] = (
        unprotected_position_errors[
            outage_end_index
        ]
    )


    protected_end_outage_errors[
        run_index
    ] = (
        protected_position_errors[
            outage_end_index
        ]
    )


    unprotected_end_outage_sigmas[
        run_index
    ] = (
        unprotected_position_sigmas[
            outage_end_index
        ]
    )


    protected_end_outage_sigmas[
        run_index
    ] = (
        protected_position_sigmas[
            outage_end_index
        ]
    )


    valid_unprotected_nis = np.isfinite(
        unprotected_nis
    )


    valid_protected_nis = np.isfinite(
        protected_nis
    )


    unprotected_nis_means[
        run_index
    ] = np.mean(
        unprotected_nis[
            valid_unprotected_nis
        ]
    )


    protected_nis_means[
        run_index
    ] = np.mean(
        protected_nis[
            valid_protected_nis
        ]
    )


    number_of_fault_epochs = np.sum(
        actual_fault_mask
    )


    detection_rates[
        run_index
    ] = (
        100.0
        * np.sum(
            fault_detected
            &
            actual_fault_mask
        )
        / number_of_fault_epochs
    )


    isolation_rates[
        run_index
    ] = (
        100.0
        * np.sum(
            correct_isolation
            &
            actual_fault_mask
        )
        / number_of_fault_epochs
    )


    nominal_test_mask = (
        fdir_attempted
        &
        ~actual_fault_mask
    )


    false_alarm_rates[
        run_index
    ] = (
        100.0
        * np.sum(
            fault_detected
            &
            nominal_test_mask
        )
        / np.sum(
            nominal_test_mask
        )
    )


    print(
        f"Run "
        f"{run_index + 1:02d}"
        f"/"
        f"{number_of_runs:02d} "
        f"termine."
    )


# ------------------------------------------------------------
# 15. STATISTIQUES ENSEMBLE
# ------------------------------------------------------------

mean_detection_rate = np.mean(
    detection_rates
)

mean_isolation_rate = np.mean(
    isolation_rates
)

mean_false_alarm_rate = np.mean(
    false_alarm_rates
)


unprotected_fault_rmse = np.sqrt(
    np.mean(
        unprotected_fault_rmse_runs**2
    )
)


protected_fault_rmse = np.sqrt(
    np.mean(
        protected_fault_rmse_runs**2
    )
)


unprotected_outage_rmse = np.sqrt(
    np.mean(
        unprotected_outage_rmse_runs**2
    )
)


protected_outage_rmse = np.sqrt(
    np.mean(
        protected_outage_rmse_runs**2
    )
)


unprotected_end_outage_rmse = np.sqrt(
    np.mean(
        unprotected_end_outage_errors**2
    )
)


protected_end_outage_rmse = np.sqrt(
    np.mean(
        protected_end_outage_errors**2
    )
)


mean_unprotected_end_sigma = np.mean(
    unprotected_end_outage_sigmas
)


mean_protected_end_sigma = np.mean(
    protected_end_outage_sigmas
)


mean_unprotected_nis = np.mean(
    unprotected_nis_means
)


mean_protected_nis = np.mean(
    protected_nis_means
)


# ------------------------------------------------------------
# 16. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================="
)

print(
    "AURORA — Experience 009-C"
)

print(
    "Monte Carlo du systeme GNSS + FDIR + EKF 9D"
)

print(
    "===================================================================================================="
)

print(
    f"Nombre de runs : "
    f"{number_of_runs}"
)

print(
    f"Satellite fautif : "
    f"{faulty_satellite_id}"
)

print(
    f"Faute pseudorange : "
    f"+{fault_bias_meters:.1f} m"
)

print(
    f"Erreur synchronisation : "
    f"{maximum_time_error:.6e} s"
)

print()

print(
    "----- FDI / FDIR -----"
)

print(
    f"Taux detection moyen : "
    f"{mean_detection_rate:.2f} %"
)

print(
    f"Taux isolation correcte moyen : "
    f"{mean_isolation_rate:.2f} %"
)

print(
    f"Taux fausse alarme moyen : "
    f"{mean_false_alarm_rate:.2f} %"
)

print()

print(
    "----- PERIODE DE FAUTE -----"
)

print(
    f"RMSE EKF non protege : "
    f"{unprotected_fault_rmse:.3f} m"
)

print(
    f"RMSE EKF protege : "
    f"{protected_fault_rmse:.3f} m"
)

print(
    f"Facteur amelioration : "
    f"{unprotected_fault_rmse / protected_fault_rmse:.2f}"
)

print()

print(
    "----- COUPURE GNSS APRES LA FAUTE -----"
)

print(
    f"RMSE outage non protege : "
    f"{unprotected_outage_rmse:.3f} m"
)

print(
    f"RMSE outage protege : "
    f"{protected_outage_rmse:.3f} m"
)

print()

print(
    f"RMSE fin outage non protege : "
    f"{unprotected_end_outage_rmse:.3f} m"
)

print(
    f"Sigma 3D moyen fin outage non protege : "
    f"{mean_unprotected_end_sigma:.3f} m"
)

print()

print(
    f"RMSE fin outage protege : "
    f"{protected_end_outage_rmse:.3f} m"
)

print(
    f"Sigma 3D moyen fin outage protege : "
    f"{mean_protected_end_sigma:.3f} m"
)

print()

print(
    "----- NIS -----"
)

print(
    f"NIS moyen non protege : "
    f"{mean_unprotected_nis:.3f}"
)

print(
    f"NIS moyen protege : "
    f"{mean_protected_nis:.3f}"
)

print(
    "Valeur theorique attendue ~3"
)

print(
    "===================================================================================================="
)