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

from src.attitude.reference_attitude import (
    build_nadir_pointing_body_to_eci,
    body_to_eci,
    eci_to_body,
    rotation_matrix_diagnostics
)

from src.sensors.body_frame_accelerometer import (
    simulate_body_frame_accelerometer_measurement
)


# ============================================================
# AURORA
# Experience 010-A
#
# Accelerometre dans le repere corps
# avec attitude nadir-pointing connue.
#
# Objectifs :
#
# 1. Construire R_BI
# 2. Verifier R^T R = I
# 3. Verifier det(R) = +1
# 4. Verifier ECI -> BODY -> ECI
# 5. Simuler biais + bruit dans les axes du capteur
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
    100.0
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
# 3. ETAT INITIAL
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
# 4. VERITE ORBITALE
# ------------------------------------------------------------

true_non_gravitational_acceleration = (
    2.0e-5
)


truth_solution = (
    propagate_orbit_with_j2_and_drag_like(
        initial_state=
            true_initial_state,

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
# 5. MODELE ACCELEROMETRE
# ------------------------------------------------------------

accelerometer_noise_std = (
    5.0e-7
)


# IMPORTANT :
#
# Le biais est maintenant defini dans
# le repere physique du capteur.
#
# Il reste donc CONSTANT dans BODY.
#
true_accelerometer_bias_body = np.array([
    1.0e-6,
    -0.8e-6,
    0.6e-6
])


accelerometer_rng = np.random.default_rng(
    4242
)


# ------------------------------------------------------------
# 6. STOCKAGE
# ------------------------------------------------------------

true_force_eci_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


true_force_body_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


measurement_body_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


measurement_eci_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


bias_eci_history = np.zeros(
    (
        number_of_epochs,
        3
    )
)


roundtrip_errors = np.zeros(
    number_of_epochs
)


orthogonality_errors = np.zeros(
    number_of_epochs
)


determinant_errors = np.zeros(
    number_of_epochs
)


determinants = np.zeros(
    number_of_epochs
)


measurement_error_body_norm = np.zeros(
    number_of_epochs
)


measurement_error_eci_norm = np.zeros(
    number_of_epochs
)


# ------------------------------------------------------------
# 7. BOUCLE TEMPORELLE
# ------------------------------------------------------------

for index in range(
    number_of_epochs
):

    position_eci = (
        truth_states[
            index,
            0:3
        ]
    )


    velocity_eci = (
        truth_states[
            index,
            3:6
        ]
    )


    # ========================================================
    # A. ATTITUDE NADIR-POINTING
    # ========================================================

    rotation_body_to_eci = (
        build_nadir_pointing_body_to_eci(
            position_eci=
                position_eci,

            velocity_eci=
                velocity_eci
        )
    )


    # ========================================================
    # B. DIAGNOSTIC MATRICE DE ROTATION
    # ========================================================

    diagnostics = (
        rotation_matrix_diagnostics(
            rotation_body_to_eci
        )
    )


    orthogonality_errors[
        index
    ] = (
        diagnostics[
            "orthogonality_error"
        ]
    )


    determinant_errors[
        index
    ] = (
        diagnostics[
            "determinant_error"
        ]
    )


    determinants[
        index
    ] = (
        diagnostics[
            "determinant"
        ]
    )


    # ========================================================
    # C. FORCE SPECIFIQUE VRAIE EN ECI
    # ========================================================

    true_specific_force_eci = (
        synthetic_drag_like_acceleration(
            time=
                time[index],

            position=
                position_eci,

            velocity=
                velocity_eci,

            base_acceleration=
                true_non_gravitational_acceleration
        )
    )


    # ========================================================
    # D. TEST ECI -> BODY -> ECI
    # ========================================================

    test_force_body = (
        eci_to_body(
            vector_eci=
                true_specific_force_eci,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    reconstructed_force_eci = (
        body_to_eci(
            vector_body=
                test_force_body,

            rotation_body_to_eci=
                rotation_body_to_eci
        )
    )


    roundtrip_errors[
        index
    ] = np.linalg.norm(
        reconstructed_force_eci
        - true_specific_force_eci
    )


    # ========================================================
    # E. MESURE ACCELEROMETRE CORPS
    # ========================================================

    measurement_result = (
        simulate_body_frame_accelerometer_measurement(
            true_specific_force_eci=
                true_specific_force_eci,

            rotation_body_to_eci=
                rotation_body_to_eci,

            bias_body=
                true_accelerometer_bias_body,

            noise_std=
                accelerometer_noise_std,

            rng=
                accelerometer_rng
        )
    )


    # ========================================================
    # F. STOCKAGE
    # ========================================================

    true_force_eci_history[
        index
    ] = (
        true_specific_force_eci
    )


    true_force_body_history[
        index
    ] = (
        measurement_result[
            "true_specific_force_body"
        ]
    )


    measurement_body_history[
        index
    ] = (
        measurement_result[
            "measurement_body"
        ]
    )


    measurement_eci_history[
        index
    ] = (
        measurement_result[
            "measurement_eci"
        ]
    )


    bias_eci_history[
        index
    ] = (
        measurement_result[
            "bias_eci"
        ]
    )


    measurement_error_body_norm[
        index
    ] = np.linalg.norm(
        measurement_result[
            "measurement_body"
        ]
        -
        measurement_result[
            "true_specific_force_body"
        ]
    )


    measurement_error_eci_norm[
        index
    ] = np.linalg.norm(
        measurement_result[
            "measurement_eci"
        ]
        -
        true_specific_force_eci
    )


# ------------------------------------------------------------
# 8. STATISTIQUES
# ------------------------------------------------------------

maximum_orthogonality_error = np.max(
    orthogonality_errors
)


maximum_determinant_error = np.max(
    determinant_errors
)


minimum_determinant = np.min(
    determinants
)


maximum_determinant = np.max(
    determinants
)


maximum_roundtrip_error = np.max(
    roundtrip_errors
)


rms_body_measurement_error = np.sqrt(
    np.mean(
        measurement_error_body_norm**2
    )
)


rms_eci_measurement_error = np.sqrt(
    np.mean(
        measurement_error_eci_norm**2
    )
)


bias_norm = np.linalg.norm(
    true_accelerometer_bias_body
)


theoretical_measurement_error_rms = np.sqrt(
    bias_norm**2
    +
    3.0
    * accelerometer_noise_std**2
)


body_bias_norm_history = np.full(
    number_of_epochs,
    bias_norm
)


eci_bias_norm_history = np.linalg.norm(
    bias_eci_history,
    axis=1
)


maximum_bias_norm_difference = np.max(
    np.abs(
        eci_bias_norm_history
        -
        body_bias_norm_history
    )
)


# ------------------------------------------------------------
# 9. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "============================================================================================"
)


print(
    "AURORA — Experience 010-A"
)


print(
    "Accelerometre repere corps + attitude nadir-pointing"
)


print(
    "============================================================================================"
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
    "----- MATRICE D'ATTITUDE -----"
)


print(
    f"Erreur orthogonalite maximale : "
    f"{maximum_orthogonality_error:.6e}"
)


print(
    f"Erreur determinant maximale : "
    f"{maximum_determinant_error:.6e}"
)


print(
    f"Determinant min : "
    f"{minimum_determinant:.12f}"
)


print(
    f"Determinant max : "
    f"{maximum_determinant:.12f}"
)


print()


print(
    "----- TRANSFORMATION DE REPERE -----"
)


print(
    f"Erreur maximale ECI -> BODY -> ECI : "
    f"{maximum_roundtrip_error:.6e} m/s^2"
)


print()


print(
    "----- ACCELEROMETRE -----"
)


print(
    f"Norme biais BODY : "
    f"{bias_norm:.6e} m/s^2"
)


print(
    f"Bruit sigma par axe : "
    f"{accelerometer_noise_std:.6e} m/s^2"
)


print(
    f"RMS theorique erreur mesure 3D : "
    f"{theoretical_measurement_error_rms:.6e} m/s^2"
)


print(
    f"RMS mesure dans BODY : "
    f"{rms_body_measurement_error:.6e} m/s^2"
)


print(
    f"RMS mesure transformee en ECI : "
    f"{rms_eci_measurement_error:.6e} m/s^2"
)


print(
    f"Variation maximale de la norme du biais "
    f"apres rotation : "
    f"{maximum_bias_norm_difference:.6e} m/s^2"
)


print(
    "============================================================================================"
)


# ------------------------------------------------------------
# 10. FIGURE FORCE SPECIFIQUE BODY
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    true_force_body_history[
        :,
        0
    ],
    label="f_x BODY vraie"
)


plt.plot(
    time_minutes,
    true_force_body_history[
        :,
        1
    ],
    label="f_y BODY vraie"
)


plt.plot(
    time_minutes,
    true_force_body_history[
        :,
        2
    ],
    label="f_z BODY vraie"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Force specifique [m/s²]"
)


plt.title(
    "AURORA — Force specifique dans le repere corps"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 11. FIGURE BIAIS BODY -> ECI
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.plot(
    time_minutes,
    bias_eci_history[
        :,
        0
    ],
    label="b_x ECI"
)


plt.plot(
    time_minutes,
    bias_eci_history[
        :,
        1
    ],
    label="b_y ECI"
)


plt.plot(
    time_minutes,
    bias_eci_history[
        :,
        2
    ],
    label="b_z ECI"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Biais transforme [m/s²]"
)


plt.title(
    "AURORA — Un biais constant BODY devient variable en ECI"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 12. FIGURE ERREUR DE TRANSFORMATION
# ------------------------------------------------------------

plt.figure(
    figsize=(12, 5)
)


plt.semilogy(
    time_minutes,
    roundtrip_errors
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Erreur round-trip [m/s²]"
)


plt.title(
    "AURORA — Validation ECI -> BODY -> ECI"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()