import numpy as np
import matplotlib.pyplot as plt

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
# Experience 009-B
#
# Integration :
#
# GNSS pseudoranges
#       ↓
# faute GNSS
#       ↓
# FDI / isolation / reconfiguration
#       ↓
# position GNSS
#       ↓
# EKF 9D + accelerometre
#
# Comparaison :
#
# 1. EKF non protege
# 2. EKF protege par FDIR
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE
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
# 2. TEMPS
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
    time[
        1
    ]
    - time[
        0
    ]
)


time_minutes = (
    time
    / 60.0
)


# ------------------------------------------------------------
# 3. FENETRE DE FAUTE
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
# 4. COUPURE GNSS COMPLETE
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
# 5. VERITE ORBITALE
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
# 6. ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


true_accelerometer_bias = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])


accelerometer_rng = (
    np.random.default_rng(
        4242
    )
)


accelerometer_measurements = np.zeros(
    (
        number_of_epochs,
        3
    )
)


for index in range(
    number_of_epochs
):

    true_specific_force = (
        synthetic_drag_like_acceleration(
            time=
                time[
                    index
                ],

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


    measurement_result = (
        simulate_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force,

            bias_eci=
                true_accelerometer_bias,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    accelerometer_measurements[
        index
    ] = (
        measurement_result[
            "measurement"
        ]
    )


# ------------------------------------------------------------
# 7. GNSS
# ------------------------------------------------------------

pseudorange_noise_std = (
    3.0
)


receiver_clock_bias_seconds = (
    100.0e-6
)


gnss_rng = np.random.default_rng(
    2026
)


fdir_confidence = (
    0.99
)


# ------------------------------------------------------------
# 8. CHOIX AUTOMATIQUE DU SATELLITE FAUTIF
#
# On choisit celui qui est visible le plus
# souvent pendant la fenetre de faute.
# ------------------------------------------------------------

visibility_counter = {
    f"GPS{index:02d}": 0
    for index in range(
        1,
        25
    )
}


fault_window_indices = np.where(
    fault_window_mask
)[0]


for index in fault_window_indices:

    (
        all_positions,
        all_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[
                index
            ]
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
        ] += (
            1
        )


faulty_satellite_id = max(
    visibility_counter,
    key=
        visibility_counter.get
)


faulty_satellite_visibility_epochs = (
    visibility_counter[
        faulty_satellite_id
    ]
)


faulty_satellite_visibility_percentage = (
    100.0
    * faulty_satellite_visibility_epochs
    / len(
        fault_window_indices
    )
)


# ------------------------------------------------------------
# 9. ETATS INITIAUX EKF
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


initial_bias_estimate = np.zeros(
    3
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

        initial_bias_estimate
    )
)


unprotected_state = (
    initial_augmented_state.copy()
)


protected_state = (
    initial_augmented_state.copy()
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

initial_bias_sigma = (
    5.0e-6
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


unprotected_covariance = (
    P0.copy()
)


protected_covariance = (
    P0.copy()
)


bias_random_walk_density = (
    1.0e-10
)


# ------------------------------------------------------------
# 11. BIAIS HORLOGE ESTIMES
# ------------------------------------------------------------

unprotected_clock_bias_meters = (
    0.0
)

protected_clock_bias_meters = (
    0.0
)


# ------------------------------------------------------------
# 12. STOCKAGE
# ------------------------------------------------------------

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


unprotected_ls_errors = np.full(
    number_of_epochs,
    np.nan
)

protected_ls_errors = np.full(
    number_of_epochs,
    np.nan
)


fault_active = np.zeros(
    number_of_epochs,
    dtype=bool
)


fdir_test_attempted = np.zeros(
    number_of_epochs,
    dtype=bool
)


fault_detected = np.zeros(
    number_of_epochs,
    dtype=bool
)


reconfigured = np.zeros(
    number_of_epochs,
    dtype=bool
)


correct_isolation = np.zeros(
    number_of_epochs,
    dtype=bool
)


isolated_ids = np.full(
    number_of_epochs,
    "",
    dtype=object
)


visible_satellite_counts = np.zeros(
    number_of_epochs,
    dtype=int
)


# ------------------------------------------------------------
# 13. ETAT INITIAL
# ------------------------------------------------------------

true_initial_augmented_state = np.concatenate(
    (
        truth_states[
            0
        ],

        true_accelerometer_bias
    )
)


initial_unprotected_error = (
    unprotected_state
    - true_initial_augmented_state
)


initial_protected_error = (
    protected_state
    - true_initial_augmented_state
)


unprotected_position_errors[
    0
] = np.linalg.norm(
    initial_unprotected_error[
        0:3
    ]
)


protected_position_errors[
    0
] = np.linalg.norm(
    initial_protected_error[
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


# ------------------------------------------------------------
# 14. BOUCLE TEMPORELLE
# ------------------------------------------------------------

for index in range(
    1,
    number_of_epochs
):

    # ========================================================
    # A. PREDICTION DES DEUX FILTRES
    # ========================================================

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


    # ========================================================
    # B. CONSTELLATION / VISIBILITE
    # ========================================================

    (
        all_gnss_positions,
        all_gnss_ids
    ) = generate_simplified_gps_positions(
        time_seconds=
            time[
                index
            ]
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
            all_gnss_positions,

        satellite_ids=
            all_gnss_ids
    )


    visible_satellite_counts[
        index
    ] = len(
        visible_positions
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


    # ========================================================
    # C. PSEUDODISTANCES COMMUNES AUX DEUX FILTRES
    # ========================================================

    if can_use_gnss:

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
                    gnss_rng
            )
        )


        pseudoranges = (
            pseudorange_result[
                "pseudoranges"
            ].copy()
        )


        # ====================================================
        # D. INJECTION DE LA FAUTE
        # ====================================================

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


        # ====================================================
        # E. FILTRE NON PROTEGE
        # ====================================================

        try:

            unprotected_gnss_result = (
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


            if unprotected_gnss_result[
                "converged"
            ]:

                unprotected_gnss_position = (
                    unprotected_gnss_result[
                        "position"
                    ]
                )


                unprotected_clock_bias_meters = (
                    unprotected_gnss_result[
                        "clock_bias_meters"
                    ]
                )


                unprotected_ls_errors[
                    index
                ] = np.linalg.norm(
                    unprotected_gnss_position
                    - truth_states[
                        index,
                        0:3
                    ]
                )


                unprotected_covariance_result = (
                    compute_gnss_solution_covariance(
                        receiver_position=
                            unprotected_gnss_position,

                        satellite_positions=
                            visible_positions,

                        pseudorange_noise_std=
                            pseudorange_noise_std
                    )
                )


                unprotected_R = (
                    unprotected_covariance_result[
                        "position_covariance"
                    ]
                )


                unprotected_update = (
                    ekf_update_position_augmented(
                        predicted_state=
                            predicted_unprotected_state,

                        predicted_covariance=
                            predicted_unprotected_covariance,

                        position_measurement=
                            unprotected_gnss_position,

                        measurement_noise_covariance=
                            unprotected_R
                    )
                )


                unprotected_innovation = (
                    unprotected_update[
                        "innovation"
                    ]
                )


                unprotected_S = (
                    unprotected_update[
                        "innovation_covariance"
                    ]
                )


                unprotected_nis[
                    index
                ] = (
                    unprotected_innovation.T
                    @ np.linalg.solve(
                        unprotected_S,
                        unprotected_innovation
                    )
                )


                unprotected_state = (
                    unprotected_update[
                        "state"
                    ]
                )


                unprotected_covariance = (
                    unprotected_update[
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


        # ====================================================
        # F. FILTRE PROTEGE PAR FDIR
        # ====================================================

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


            fdir_test_attempted[
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


            reconfigured[
                index
            ] = (
                fdir_result[
                    "reconfigured"
                ]
            )


            isolated_id = (
                fdir_result.get(
                    "isolated_id",
                    None
                )
            )


            if isolated_id is not None:

                isolated_ids[
                    index
                ] = (
                    isolated_id
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

                protected_gnss_result = (
                    fdir_result[
                        "solution"
                    ]
                )


                protected_positions_used = (
                    fdir_result[
                        "satellite_positions"
                    ]
                )


                protected_gnss_position = (
                    protected_gnss_result[
                        "position"
                    ]
                )


                protected_clock_bias_meters = (
                    protected_gnss_result[
                        "clock_bias_meters"
                    ]
                )


                protected_ls_errors[
                    index
                ] = np.linalg.norm(
                    protected_gnss_position
                    - truth_states[
                        index,
                        0:3
                    ]
                )


                protected_covariance_result = (
                    compute_gnss_solution_covariance(
                        receiver_position=
                            protected_gnss_position,

                        satellite_positions=
                            protected_positions_used,

                        pseudorange_noise_std=
                            pseudorange_noise_std
                    )
                )


                protected_R = (
                    protected_covariance_result[
                        "position_covariance"
                    ]
                )


                protected_update = (
                    ekf_update_position_augmented(
                        predicted_state=
                            predicted_protected_state,

                        predicted_covariance=
                            predicted_protected_covariance,

                        position_measurement=
                            protected_gnss_position,

                        measurement_noise_covariance=
                            protected_R
                    )
                )


                protected_innovation = (
                    protected_update[
                        "innovation"
                    ]
                )


                protected_S = (
                    protected_update[
                        "innovation_covariance"
                    ]
                )


                protected_nis[
                    index
                ] = (
                    protected_innovation.T
                    @ np.linalg.solve(
                        protected_S,
                        protected_innovation
                    )
                )


                protected_state = (
                    protected_update[
                        "state"
                    ]
                )


                protected_covariance = (
                    protected_update[
                        "covariance"
                    ]
                )


            else:

                # Detection sans isolation fiable :
                # on prefere ignorer le GNSS a cette epoque.

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


    # ========================================================
    # G. COUPURE GNSS
    # ========================================================

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


    # ========================================================
    # H. ERREURS DES DEUX FILTRES
    # ========================================================

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


# ------------------------------------------------------------
# 15. MASQUES D'EVALUATION
# ------------------------------------------------------------

evaluation_mask = np.ones(
    number_of_epochs,
    dtype=bool
)

evaluation_mask[
    0
] = False


healthy_gnss_mask = (
    forced_gnss_available
    &
    ~fault_active
    &
    evaluation_mask
)


actual_fault_mask = (
    fault_active
)


valid_unprotected_nis = np.isfinite(
    unprotected_nis
)


valid_protected_nis = np.isfinite(
    protected_nis
)


valid_unprotected_ls = np.isfinite(
    unprotected_ls_errors
)


valid_protected_ls = np.isfinite(
    protected_ls_errors
)


# ------------------------------------------------------------
# 16. FDI / FDIR STATISTIQUES
# ------------------------------------------------------------

number_of_fault_epochs = np.sum(
    actual_fault_mask
)


number_of_detected_fault_epochs = np.sum(
    fault_detected
    &
    actual_fault_mask
)


number_of_correct_isolations = np.sum(
    correct_isolation
    &
    actual_fault_mask
)


detection_rate = (
    100.0
    * number_of_detected_fault_epochs
    / number_of_fault_epochs
)


correct_isolation_rate = (
    100.0
    * number_of_correct_isolations
    / number_of_fault_epochs
)


if number_of_detected_fault_epochs > 0:

    conditional_isolation_rate = (
        100.0
        * number_of_correct_isolations
        / number_of_detected_fault_epochs
    )

else:

    conditional_isolation_rate = (
        np.nan
    )


nominal_test_mask = (
    fdir_test_attempted
    &
    ~actual_fault_mask
)


number_of_nominal_tests = np.sum(
    nominal_test_mask
)


number_of_false_alarms = np.sum(
    fault_detected
    &
    nominal_test_mask
)


false_alarm_rate = (
    100.0
    * number_of_false_alarms
    / number_of_nominal_tests
)


# ------------------------------------------------------------
# 17. RMSE PENDANT FAUTE
# ------------------------------------------------------------

unprotected_rmse_fault = np.sqrt(
    np.mean(
        unprotected_position_errors[
            actual_fault_mask
        ]**2
    )
)


protected_rmse_fault = np.sqrt(
    np.mean(
        protected_position_errors[
            actual_fault_mask
        ]**2
    )
)


unprotected_ls_rmse_fault = np.sqrt(
    np.mean(
        unprotected_ls_errors[
            actual_fault_mask
            &
            valid_unprotected_ls
        ]**2
    )
)


protected_ls_rmse_fault = np.sqrt(
    np.mean(
        protected_ls_errors[
            actual_fault_mask
            &
            valid_protected_ls
        ]**2
    )
)


# ------------------------------------------------------------
# 18. RMSE NOMINAL
# ------------------------------------------------------------

unprotected_rmse_healthy = np.sqrt(
    np.mean(
        unprotected_position_errors[
            healthy_gnss_mask
        ]**2
    )
)


protected_rmse_healthy = np.sqrt(
    np.mean(
        protected_position_errors[
            healthy_gnss_mask
        ]**2
    )
)


# ------------------------------------------------------------
# 19. RMSE OUTAGE
# ------------------------------------------------------------

unprotected_rmse_outage = np.sqrt(
    np.mean(
        unprotected_position_errors[
            outage_mask
        ]**2
    )
)


protected_rmse_outage = np.sqrt(
    np.mean(
        protected_position_errors[
            outage_mask
        ]**2
    )
)


outage_indices = np.where(
    outage_mask
)[0]


outage_end_index = (
    outage_indices[
        -1
    ]
)


# ------------------------------------------------------------
# 20. NIS
# ------------------------------------------------------------

mean_unprotected_nis = np.mean(
    unprotected_nis[
        valid_unprotected_nis
    ]
)


mean_protected_nis = np.mean(
    protected_nis[
        valid_protected_nis
    ]
)


unprotected_fault_nis = np.mean(
    unprotected_nis[
        actual_fault_mask
        &
        valid_unprotected_nis
    ]
)


protected_fault_nis = np.mean(
    protected_nis[
        actual_fault_mask
        &
        valid_protected_nis
    ]
)


# ------------------------------------------------------------
# 21. RETOUR APRES OUTAGE
# ------------------------------------------------------------

after_outage_indices = np.where(
    time_minutes
    > outage_end_minutes
)[0]


first_return_index = (
    after_outage_indices[
        0
    ]
)


# ------------------------------------------------------------
# 22. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "===================================================================================================="
)


print(
    "AURORA — Experience 009-B"
)


print(
    "Integration FDIR GNSS + EKF 9D"
)


print(
    "===================================================================================================="
)


print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
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
    "----- SCENARIO -----"
)


print(
    f"Satellite fautif : "
    f"{faulty_satellite_id}"
)


print(
    f"Visibilite du satellite pendant fenetre faute : "
    f"{faulty_satellite_visibility_percentage:.2f} %"
)


print(
    f"Faute pseudorange : "
    f"+{fault_bias_meters:.1f} m"
)


print(
    f"Fenetre faute : "
    f"{fault_start_minutes:.1f} "
    f"a {fault_end_minutes:.1f} min"
)


print(
    f"Coupure GNSS : "
    f"{outage_start_minutes:.1f} "
    f"a {outage_end_minutes:.1f} min"
)


print()


print(
    "----- FDI / FDIR -----"
)


print(
    f"Epoques avec faute reellement active : "
    f"{number_of_fault_epochs}"
)


print(
    f"Taux de detection : "
    f"{detection_rate:.2f} %"
)


print(
    f"Taux isolation correcte total : "
    f"{correct_isolation_rate:.2f} %"
)


print(
    f"Taux isolation correcte conditionnel : "
    f"{conditional_isolation_rate:.2f} %"
)


print(
    f"Taux de fausse alarme hors faute : "
    f"{false_alarm_rate:.2f} %"
)


print()


print(
    "----- POSITION GNSS LS PENDANT FAUTE -----"
)


print(
    f"RMSE LS non protege : "
    f"{unprotected_ls_rmse_fault:.3f} m"
)


print(
    f"RMSE LS avec FDIR : "
    f"{protected_ls_rmse_fault:.3f} m"
)


print()


print(
    "----- EKF PENDANT FAUTE -----"
)


print(
    f"RMSE EKF non protege : "
    f"{unprotected_rmse_fault:.3f} m"
)


print(
    f"RMSE EKF protege : "
    f"{protected_rmse_fault:.3f} m"
)


print(
    f"NIS moyen non protege pendant faute : "
    f"{unprotected_fault_nis:.3f}"
)


print(
    f"NIS moyen protege pendant faute : "
    f"{protected_fault_nis:.3f}"
)


print()


print(
    "----- PERIODE GNSS SAINE -----"
)


print(
    f"RMSE EKF non protege : "
    f"{unprotected_rmse_healthy:.3f} m"
)


print(
    f"RMSE EKF protege : "
    f"{protected_rmse_healthy:.3f} m"
)


print()


print(
    "----- COUPURE GNSS -----"
)


print(
    f"RMSE outage non protege : "
    f"{unprotected_rmse_outage:.3f} m"
)


print(
    f"RMSE outage protege : "
    f"{protected_rmse_outage:.3f} m"
)


print(
    f"Erreur fin outage non protege : "
    f"{unprotected_position_errors[outage_end_index]:.3f} m"
)


print(
    f"Erreur fin outage protege : "
    f"{protected_position_errors[outage_end_index]:.3f} m"
)


print(
    f"Sigma fin outage non protege : "
    f"{unprotected_position_sigmas[outage_end_index]:.3f} m"
)


print(
    f"Sigma fin outage protege : "
    f"{protected_position_sigmas[outage_end_index]:.3f} m"
)


print()


print(
    "----- RETOUR GNSS -----"
)


print(
    f"Erreur premiere correction non protegee : "
    f"{unprotected_position_errors[first_return_index]:.3f} m"
)


print(
    f"Erreur premiere correction protegee : "
    f"{protected_position_errors[first_return_index]:.3f} m"
)


print()


print(
    "----- NIS GLOBAL -----"
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
    "Valeur nominale attendue ~3"
)


print(
    "===================================================================================================="
)


# ------------------------------------------------------------
# 23. FIGURE ERREUR POSITION
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    unprotected_position_errors,
    label="EKF non protege"
)


plt.plot(
    time_minutes,
    protected_position_errors,
    label="EKF protege FDIR"
)


plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15,
    label="Faute GNSS"
)


plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15,
    label="Coupure GNSS"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur position 3D [m]"
)


plt.title(
    "AURORA — Effet du FDIR sur la navigation recursive"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 24. FIGURE GNSS LS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    unprotected_ls_errors,
    label="GNSS LS non protege"
)


plt.plot(
    time_minutes,
    protected_ls_errors,
    label="GNSS LS apres FDIR"
)


plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur position GNSS [m]"
)


plt.title(
    "AURORA — Protection de la solution GNSS par FDIR"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 25. FIGURE DETECTION
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 4)
)


plt.step(
    time_minutes,
    fault_active.astype(
        int
    ),
    where="post",
    label="Faute active"
)


plt.step(
    time_minutes,
    fault_detected.astype(
        int
    ),
    where="post",
    label="Faute detectee"
)


plt.step(
    time_minutes,
    correct_isolation.astype(
        int
    ),
    where="post",
    label="Isolation correcte"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Etat logique"
)


plt.yticks(
    [
        0,
        1
    ]
)


plt.title(
    "AURORA — Detection et isolation temporelles"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 26. FIGURE NIS
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 5)
)


plt.plot(
    time_minutes,
    unprotected_nis,
    label="NIS non protege"
)


plt.plot(
    time_minutes,
    protected_nis,
    label="NIS protege"
)


plt.axhline(
    3.0,
    linestyle="--",
    label="Moyenne theorique = 3"
)


plt.axvspan(
    fault_start_minutes,
    fault_end_minutes,
    alpha=0.15
)


plt.axvspan(
    outage_start_minutes,
    outage_end_minutes,
    alpha=0.15
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "NIS [-]"
)


plt.title(
    "AURORA — Innovations EKF avec et sans protection FDIR"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()