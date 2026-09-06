import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.orbit import (
    R_EARTH,
    propagate_orbit
)

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian
)

from src.navigation.gnss_constellation import (
    generate_simplified_gps_constellation,
    propagate_gnss_constellation
)

from src.navigation.gnss_visibility import (
    is_gnss_visible
)

from src.navigation.gnss_measurements import (
    C_LIGHT,
    simulate_pseudorange_set
)

from src.navigation.gnss_positioning import (
    solve_position_least_squares,
    build_pseudorange_jacobian
)

from src.faults.gnss_faults import (
    inject_pseudorange_bias
)

from src.faults.gnss_fdi import (
    global_residual_test,
    compute_normalized_residuals,
    isolate_fault_leave_one_out
)


# ============================================================
# AURORA
# Expérience 006-B
# Détection, isolation et reconfiguration GNSS
# ============================================================


# ------------------------------------------------------------
# 1. ORBITE DU CUBESAT
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

argument_of_periapsis = (
    np.deg2rad(30.0)
)

true_anomaly = (
    np.deg2rad(25.0)
)


# ------------------------------------------------------------
# 2. PARAMETRES GNSS / FDI
# ------------------------------------------------------------

measurement_time = (
    120.0 * 60.0
)

receiver_clock_bias_seconds = (
    100.0e-6
)

measurement_noise_std = 3.0

fault_bias_meters = 100.0

confidence_level = 0.99

random_seed = 42


# ------------------------------------------------------------
# 3. ETAT VRAI DU CUBESAT
# ------------------------------------------------------------

initial_position, initial_velocity = (
    keplerian_to_cartesian(
        semi_major_axis,
        eccentricity,
        inclination,
        raan,
        argument_of_periapsis,
        true_anomaly
    )
)

initial_state = np.concatenate(
    (
        initial_position,
        initial_velocity
    )
)

cub_sat_solution = propagate_orbit(
    initial_state=
        initial_state,

    duration=
        measurement_time,

    number_of_points=
        2
)

true_receiver_position = (
    cub_sat_solution.y[
        0:3,
        -1
    ]
)


# ------------------------------------------------------------
# 4. CONSTELLATION GPS
# ------------------------------------------------------------

gps_constellation = (
    generate_simplified_gps_constellation()
)

gps_solutions = (
    propagate_gnss_constellation(
        constellation=
            gps_constellation,

        duration=
            measurement_time,

        number_of_points=
            2
    )
)


# ------------------------------------------------------------
# 5. SATELLITES VISIBLES
# ------------------------------------------------------------

visible_positions = []

visible_ids = []


for satellite, solution in zip(
    gps_constellation,
    gps_solutions
):

    satellite_position = (
        solution.y[
            0:3,
            -1
        ]
    )

    if is_gnss_visible(
        true_receiver_position,
        satellite_position
    ):

        visible_positions.append(
            satellite_position
        )

        visible_ids.append(
            satellite[
                "id"
            ]
        )


visible_positions = np.array(
    visible_positions
)


# ------------------------------------------------------------
# 6. PSEUDORANGES NOMINALES
# ------------------------------------------------------------

measurements = simulate_pseudorange_set(
    receiver_position=
        true_receiver_position,

    satellite_positions=
        visible_positions,

    receiver_clock_bias_seconds=
        receiver_clock_bias_seconds,

    noise_std=
        measurement_noise_std,

    random_seed=
        random_seed
)

nominal_pseudoranges = (
    measurements[
        "pseudoranges"
    ]
)


# ------------------------------------------------------------
# 7. INJECTION DE LA FAUTE
# ------------------------------------------------------------

faulty_satellite_index = 0

faulty_satellite_id = (
    visible_ids[
        faulty_satellite_index
    ]
)

faulty_pseudoranges = (
    inject_pseudorange_bias(
        pseudoranges=
            nominal_pseudoranges,

        satellite_index=
            faulty_satellite_index,

        bias_meters=
            fault_bias_meters
    )
)


# ------------------------------------------------------------
# 8. ESTIMATION INITIALE
# ------------------------------------------------------------

initial_position_guess = (
    true_receiver_position
    + np.array([
        20_000.0,
        -15_000.0,
        10_000.0
    ])
)


# ------------------------------------------------------------
# 9. FONCTION DE SOLUTION + TEST
# ------------------------------------------------------------

def solve_and_test(
    pseudoranges
):

    solution = (
        solve_position_least_squares(
            satellite_positions=
                visible_positions,

            pseudoranges=
                pseudoranges,

            initial_position=
                initial_position_guess,

            initial_clock_bias_range=
                0.0,

            tolerance=
                1e-4,

            max_iterations=
                20
        )
    )

    H = build_pseudorange_jacobian(
        receiver_position=
            solution[
                "position"
            ],

        satellite_positions=
            visible_positions
    )

    rank = np.linalg.matrix_rank(
        H
    )

    degrees_of_freedom = (
        len(
            pseudoranges
        )
        - rank
    )

    global_test = (
        global_residual_test(
            residuals=
                solution[
                    "final_residuals"
                ],

            measurement_noise_std=
                measurement_noise_std,

            degrees_of_freedom=
                degrees_of_freedom,

            confidence_level=
                confidence_level
        )
    )

    normalized_residuals = (
        compute_normalized_residuals(
            receiver_position=
                solution[
                    "position"
                ],

            satellite_positions=
                visible_positions,

            residuals=
                solution[
                    "final_residuals"
                ],

            measurement_noise_std=
                measurement_noise_std
        )
    )

    position_error = np.linalg.norm(
        solution[
            "position"
        ]
        - true_receiver_position
    )

    return (
        solution,
        global_test,
        normalized_residuals,
        position_error,
        degrees_of_freedom
    )


# ------------------------------------------------------------
# 10. CAS NOMINAL
# ------------------------------------------------------------

(
    nominal_solution,
    nominal_test,
    nominal_normalized_residuals,
    nominal_position_error,
    nominal_dof
) = solve_and_test(
    nominal_pseudoranges
)


# ------------------------------------------------------------
# 11. CAS FAUTE
# ------------------------------------------------------------

(
    faulty_solution,
    faulty_test,
    faulty_normalized_residuals,
    faulty_position_error,
    faulty_dof
) = solve_and_test(
    faulty_pseudoranges
)


# ------------------------------------------------------------
# 12. PREMIER INDICE D'ISOLATION :
# PLUS GRAND RESIDU NORMALISE
# ------------------------------------------------------------

largest_residual_index = np.argmax(
    np.abs(
        faulty_normalized_residuals
    )
)

largest_residual_satellite_id = (
    visible_ids[
        largest_residual_index
    ]
)


# ------------------------------------------------------------
# 13. ISOLATION LEAVE-ONE-OUT
# ------------------------------------------------------------

isolation_result = (
    isolate_fault_leave_one_out(
        satellite_positions=
            visible_positions,

        pseudoranges=
            faulty_pseudoranges,

        satellite_ids=
            visible_ids,

        initial_position=
            initial_position_guess,

        initial_clock_bias_range=
            0.0,

        measurement_noise_std=
            measurement_noise_std,

        confidence_level=
            confidence_level,

        max_iterations=
            20
    )
)

suspect_index = isolation_result[
    "suspect_index"
]

suspect_id = isolation_result[
    "suspect_id"
]


# ------------------------------------------------------------
# 14. RECONFIGURATION
# ------------------------------------------------------------

if suspect_index is None:

    raise RuntimeError(
        "Aucun satellite suspect "
        "n'a pu être isolé."
    )


healthy_mask = np.ones(
    len(
        visible_ids
    ),
    dtype=bool
)

healthy_mask[
    suspect_index
] = False


reconfigured_positions = (
    visible_positions[
        healthy_mask
    ]
)

reconfigured_pseudoranges = (
    faulty_pseudoranges[
        healthy_mask
    ]
)


reconfigured_solution = (
    solve_position_least_squares(
        satellite_positions=
            reconfigured_positions,

        pseudoranges=
            reconfigured_pseudoranges,

        initial_position=
            initial_position_guess,

        initial_clock_bias_range=
            0.0,

        tolerance=
            1e-4,

        max_iterations=
            20
    )
)


reconfigured_position_error = (
    np.linalg.norm(
        reconfigured_solution[
            "position"
        ]
        - true_receiver_position
    )
)


# ------------------------------------------------------------
# 15. TEST GLOBAL APRES RECONFIGURATION
# ------------------------------------------------------------

reconfigured_H = (
    build_pseudorange_jacobian(
        receiver_position=
            reconfigured_solution[
                "position"
            ],

        satellite_positions=
            reconfigured_positions
    )
)

reconfigured_rank = (
    np.linalg.matrix_rank(
        reconfigured_H
    )
)

reconfigured_dof = (
    len(
        reconfigured_pseudoranges
    )
    - reconfigured_rank
)

reconfigured_test = (
    global_residual_test(
        residuals=
            reconfigured_solution[
                "final_residuals"
            ],

        measurement_noise_std=
            measurement_noise_std,

        degrees_of_freedom=
            reconfigured_dof,

        confidence_level=
            confidence_level
    )
)


# ------------------------------------------------------------
# 16. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=============================================================="
)

print(
    "AURORA — Expérience 006-B"
)

print(
    "Détection, isolation et reconfiguration GNSS"
)

print(
    "=============================================================="
)

print(
    f"Satellites visibles : "
    f"{len(visible_ids)}"
)

print(
    f"Satellite réellement fauté : "
    f"GPS {faulty_satellite_id:02d}"
)

print(
    f"Biais injecté : "
    f"{fault_bias_meters:.1f} m"
)

print(
    f"Niveau de confiance : "
    f"{confidence_level * 100:.1f} %"
)

print()

print(
    "----- CAS NOMINAL -----"
)

print(
    f"Erreur position : "
    f"{nominal_position_error:.3f} m"
)

print(
    f"RMS résidus : "
    f"{nominal_solution['final_residual_rms']:.3f} m"
)

print(
    f"Statistique T : "
    f"{nominal_test['statistic']:.3f}"
)

print(
    f"Seuil Chi2 : "
    f"{nominal_test['threshold']:.3f}"
)

print(
    f"Faute détectée : "
    f"{nominal_test['detected']}"
)

print()

print(
    "----- CAS FAUTE -----"
)

print(
    f"Erreur position avant FDI : "
    f"{faulty_position_error:.3f} m"
)

print(
    f"RMS résidus : "
    f"{faulty_solution['final_residual_rms']:.3f} m"
)

print(
    f"Statistique T : "
    f"{faulty_test['statistic']:.3f}"
)

print(
    f"Seuil Chi2 : "
    f"{faulty_test['threshold']:.3f}"
)

print(
    f"Faute détectée : "
    f"{faulty_test['detected']}"
)

print()

print(
    "Satellite avec plus grand "
    "résidu normalisé : "
    f"GPS {largest_residual_satellite_id:02d}"
)

print(
    f"Satellite isolé par leave-one-out : "
    f"GPS {suspect_id:02d}"
)

print()

print(
    "----- APRES RECONFIGURATION -----"
)

print(
    f"Satellite rejeté : "
    f"GPS {suspect_id:02d}"
)

print(
    f"Satellites restants : "
    f"{len(reconfigured_positions)}"
)

print(
    f"Erreur position après FDIR : "
    f"{reconfigured_position_error:.3f} m"
)

print(
    f"RMS résidus après FDIR : "
    f"{reconfigured_solution['final_residual_rms']:.3f} m"
)

print(
    f"Statistique T après FDIR : "
    f"{reconfigured_test['statistic']:.3f}"
)

print(
    f"Seuil Chi2 après FDIR : "
    f"{reconfigured_test['threshold']:.3f}"
)

print(
    f"Faute encore détectée : "
    f"{reconfigured_test['detected']}"
)

print(
    "=============================================================="
)


# ------------------------------------------------------------
# 17. TABLEAU DES RESIDUS NORMALISES
# ------------------------------------------------------------

print(
    "\n"
    "GPS | Résidu normalisé nominal "
    "| Résidu normalisé fauté"
)

print(
    "-----------------------------------------------"
)


for satellite_id, nominal_z, faulty_z in zip(
    visible_ids,
    nominal_normalized_residuals,
    faulty_normalized_residuals
):

    print(
        f"{satellite_id:02d}  | "
        f"{nominal_z:>11.3f}           | "
        f"{faulty_z:>11.3f}"
    )


# ------------------------------------------------------------
# 18. FIGURE RESIDUS NORMALISES
# ------------------------------------------------------------

indices = np.arange(
    len(
        visible_ids
    )
)


plt.figure(
    figsize=(11, 5)
)

plt.scatter(
    indices,
    faulty_normalized_residuals,
    label="Résidus normalisés"
)

plt.axhline(
    3.0,
    linestyle="--",
    label="+3 sigma"
)

plt.axhline(
    -3.0,
    linestyle="--",
    label="-3 sigma"
)

plt.xticks(
    indices,
    visible_ids
)

plt.xlabel(
    "Satellite GPS"
)

plt.ylabel(
    "Résidu normalisé [-]"
)

plt.title(
    "AURORA — Isolation d'une anomalie GNSS"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FIGURE AVANT / APRES FDIR
# ------------------------------------------------------------

labels = [
    "Nominal",
    "Faute",
    "Après FDIR"
]

errors = [
    nominal_position_error,
    faulty_position_error,
    reconfigured_position_error
]


plt.figure(
    figsize=(9, 5)
)

plt.bar(
    labels,
    errors
)

plt.ylabel(
    "Erreur de position 3D [m]"
)

plt.title(
    "AURORA — Effet de la reconfiguration GNSS"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()