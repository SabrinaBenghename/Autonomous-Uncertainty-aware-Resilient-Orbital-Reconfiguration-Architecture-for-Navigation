import numpy as np
import matplotlib.pyplot as plt

from src.faults.resilience_manager import (
    ResilienceManager,
    ResilienceMode
)


# ============================================================
# AURORA
# Experience 013-A
#
# Validation du gestionnaire de modes de resilience.
#
#
# Cette experience valide la LOGIQUE DE SUPERVISION
# independamment des estimateurs numeriques.
#
#
# Scenario :
#
#   0 - 20 min
#       nominal
#
#   20 - 30 min
#       faute GNSS detectee + reconfiguration validee
#
#   40 - 55 min
#       coupure complete GNSS
#
#   65 - 75 min
#       perte star tracker
#
#   85 - 95 min
#       star tracker disponible mais
#       incertitude attitude trop grande
#
#   105 - 115 min
#       faute GNSS detectee non isolable
#       -> mesure GNSS rejetee
#
#   125 - 140 min
#       coupure GNSS + perte star tracker
#
#
# Entre chaque evenement :
#
#       RECOVERY
#           puis
#       NOMINAL
# ============================================================


# ------------------------------------------------------------
# 1. TEMPS
# ------------------------------------------------------------

simulation_duration_minutes = (
    150.0
)


dt = (
    10.0
)


simulation_duration = (
    simulation_duration_minutes
    * 60.0
)


number_of_epochs = (
    int(
        simulation_duration
        / dt
    )
    + 1
)


time = np.linspace(
    0.0,
    simulation_duration,
    number_of_epochs
)


time_minutes = (
    time
    / 60.0
)


# ------------------------------------------------------------
# 2. PARAMETRES DU MODE MANAGER
# ------------------------------------------------------------

attitude_sigma_limit_deg = (
    0.5
)


attitude_sigma_limit_rad = np.deg2rad(
    attitude_sigma_limit_deg
)


# 6 epochs x 10 s = 60 s de recovery.
recovery_epochs = (
    6
)


manager = (
    ResilienceManager(
        attitude_sigma_limit_rad=
            attitude_sigma_limit_rad,

        recovery_epochs=
            recovery_epochs
    )
)


# ------------------------------------------------------------
# 3. HISTORIQUES
# ------------------------------------------------------------

mode_history = []


raw_mode_history = []


severity_history = np.zeros(
    number_of_epochs,
    dtype=int
)


gnss_used_history = np.zeros(
    number_of_epochs,
    dtype=bool
)


star_tracker_used_history = np.zeros(
    number_of_epochs,
    dtype=bool
)


attitude_sigma_history = np.zeros(
    number_of_epochs
)


# ------------------------------------------------------------
# 4. SCENARIO
# ------------------------------------------------------------

for index in range(
    number_of_epochs
):

    current_time_minutes = (
        time_minutes[
            index
        ]
    )


    # ========================================================
    # Etat nominal par defaut
    # ========================================================

    gnss_signal_available = (
        True
    )


    gnss_measurement_accepted = (
        True
    )


    gnss_fault_detected = (
        False
    )


    gnss_reconfigured = (
        False
    )


    star_tracker_available = (
        True
    )


    attitude_sigma_deg = (
        0.03
    )


    # ========================================================
    # A. FAUTE GNSS RECONFIGUREE
    # ========================================================

    if (
        20.0
        <=
        current_time_minutes
        <
        30.0
    ):

        gnss_signal_available = (
            True
        )


        gnss_measurement_accepted = (
            True
        )


        gnss_fault_detected = (
            True
        )


        gnss_reconfigured = (
            True
        )


    # ========================================================
    # B. COUPURE GNSS
    # ========================================================

    if (
        40.0
        <=
        current_time_minutes
        <
        55.0
    ):

        gnss_signal_available = (
            False
        )


        gnss_measurement_accepted = (
            False
        )


    # ========================================================
    # C. PERTE STAR TRACKER
    # ========================================================

    if (
        65.0
        <=
        current_time_minutes
        <
        75.0
    ):

        star_tracker_available = (
            False
        )


        # Le MEKF continue par propagation gyro.
        attitude_sigma_deg = (
            0.20
        )


    # ========================================================
    # D. ATTITUDE INCERTAINE
    #
    # Star tracker physiquement disponible,
    # mais diagnostic d'incertitude trop eleve.
    # ========================================================

    if (
        85.0
        <=
        current_time_minutes
        <
        95.0
    ):

        star_tracker_available = (
            True
        )


        attitude_sigma_deg = (
            0.80
        )


    # ========================================================
    # E. GNSS FAUTIF NON RECONFIGURABLE
    # ========================================================

    if (
        105.0
        <=
        current_time_minutes
        <
        115.0
    ):

        gnss_signal_available = (
            True
        )


        gnss_fault_detected = (
            True
        )


        gnss_reconfigured = (
            False
        )


        gnss_measurement_accepted = (
            False
        )


    # ========================================================
    # F. DOUBLE OUTAGE
    # ========================================================

    if (
        125.0
        <=
        current_time_minutes
        <
        140.0
    ):

        gnss_signal_available = (
            False
        )


        gnss_measurement_accepted = (
            False
        )


        star_tracker_available = (
            False
        )


        attitude_sigma_deg = (
            0.30
        )


    # ========================================================
    # UPDATE MODE MANAGER
    # ========================================================

    attitude_sigma_rad = np.deg2rad(
        attitude_sigma_deg
    )


    decision = (
        manager.update(
            gnss_signal_available=
                gnss_signal_available,

            gnss_measurement_accepted=
                gnss_measurement_accepted,

            gnss_fault_detected=
                gnss_fault_detected,

            gnss_reconfigured=
                gnss_reconfigured,

            star_tracker_available=
                star_tracker_available,

            attitude_sigma_rad=
                attitude_sigma_rad
        )
    )


    mode_history.append(
        decision.mode
    )


    raw_mode_history.append(
        decision.raw_mode
    )


    severity_history[
        index
    ] = (
        decision.severity
    )


    gnss_used_history[
        index
    ] = (
        decision.use_gnss_update
    )


    star_tracker_used_history[
        index
    ] = (
        decision.use_star_tracker_update
    )


    attitude_sigma_history[
        index
    ] = (
        attitude_sigma_deg
    )


# ------------------------------------------------------------
# 5. TRANSITIONS
# ------------------------------------------------------------

transitions = []


previous_mode = (
    mode_history[
        0
    ]
)


transitions.append(
    (
        time_minutes[
            0
        ],

        previous_mode
    )
)


for index in range(
    1,
    number_of_epochs
):

    current_mode = (
        mode_history[
            index
        ]
    )


    if (
        current_mode
        !=
        previous_mode
    ):

        transitions.append(
            (
                time_minutes[
                    index
                ],

                current_mode
            )
        )


        previous_mode = (
            current_mode
        )


# ------------------------------------------------------------
# 6. OCCUPATION DES MODES
# ------------------------------------------------------------

all_modes = [
    ResilienceMode.NOMINAL,
    ResilienceMode.GNSS_RECONFIGURED,
    ResilienceMode.GNSS_OUTAGE,
    ResilienceMode.GNSS_REJECTED,
    ResilienceMode.ATTITUDE_DEGRADED,
    ResilienceMode.DUAL_OUTAGE,
    ResilienceMode.RECOVERY
]


mode_counts = {}


mode_durations_minutes = {}


for mode in all_modes:

    count = sum(
        current_mode
        ==
        mode
        for current_mode in mode_history
    )


    mode_counts[
        mode
    ] = (
        count
    )


    mode_durations_minutes[
        mode
    ] = (
        count
        * dt
        / 60.0
    )


# ------------------------------------------------------------
# 7. VERIFICATIONS
# ------------------------------------------------------------

required_modes = [
    ResilienceMode.NOMINAL,
    ResilienceMode.GNSS_RECONFIGURED,
    ResilienceMode.GNSS_OUTAGE,
    ResilienceMode.GNSS_REJECTED,
    ResilienceMode.ATTITUDE_DEGRADED,
    ResilienceMode.DUAL_OUTAGE,
    ResilienceMode.RECOVERY
]


all_required_modes_observed = all(
    mode_counts[
        mode
    ]
    >
    0
    for mode in required_modes
)


# ------------------------------------------------------------
# FDIR reconfigure :
#
# GNSS doit rester utilise.
# ------------------------------------------------------------

reconfigured_mask = np.array([
    mode
    ==
    ResilienceMode.GNSS_RECONFIGURED
    for mode in mode_history
])


reconfigured_gnss_used = (
    np.all(
        gnss_used_history[
            reconfigured_mask
        ]
    )
)


# ------------------------------------------------------------
# GNSS rejected :
#
# GNSS ne doit jamais etre utilise.
# ------------------------------------------------------------

rejected_mask = np.array([
    mode
    ==
    ResilienceMode.GNSS_REJECTED
    for mode in mode_history
])


rejected_gnss_blocked = (
    np.all(
        ~gnss_used_history[
            rejected_mask
        ]
    )
)


# ------------------------------------------------------------
# Double outage :
#
# ni GNSS ni star tracker ne doivent etre utilises.
# ------------------------------------------------------------

dual_mask = np.array([
    mode
    ==
    ResilienceMode.DUAL_OUTAGE
    for mode in mode_history
])


dual_gnss_blocked = (
    np.all(
        ~gnss_used_history[
            dual_mask
        ]
    )
)


dual_star_tracker_blocked = (
    np.all(
        ~star_tracker_used_history[
            dual_mask
        ]
    )
)


logic_validated = (
    all_required_modes_observed
    and
    reconfigured_gnss_used
    and
    rejected_gnss_blocked
    and
    dual_gnss_blocked
    and
    dual_star_tracker_blocked
)


# ------------------------------------------------------------
# 8. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=============================================================================================================="
)


print(
    "AURORA — Experience 013-A"
)


print(
    "Validation du gestionnaire de modes de resilience"
)


print(
    "=============================================================================================================="
)


print(
    f"Duree : "
    f"{simulation_duration_minutes:.1f} min"
)


print(
    f"Pas temporel : "
    f"{dt:.1f} s"
)


print(
    f"Limite sigma attitude : "
    f"{attitude_sigma_limit_deg:.3f} deg"
)


print(
    f"Duree recovery : "
    f"{recovery_epochs * dt:.1f} s"
)


print()


print(
    "----- TRANSITIONS -----"
)


for (
    transition_time,
    transition_mode
) in transitions:

    print(
        f"t = "
        f"{transition_time:7.2f} min"
        f"  ->  "
        f"{transition_mode.value}"
    )


print()


print(
    "----- OCCUPATION MODES -----"
)


for mode in all_modes:

    print(
        f"{mode.value:22s} : "
        f"{mode_durations_minutes[mode]:7.2f} min "
        f"({mode_counts[mode]} epochs)"
    )


print()


print(
    "----- VERIFICATION LOGIQUE -----"
)


print(
    f"Tous les modes observes : "
    f"{all_required_modes_observed}"
)


print(
    f"GNSS utilise apres reconfiguration validee : "
    f"{reconfigured_gnss_used}"
)


print(
    f"GNSS bloque apres rejet FDIR : "
    f"{rejected_gnss_blocked}"
)


print(
    f"GNSS bloque pendant dual outage : "
    f"{dual_gnss_blocked}"
)


print(
    f"Star tracker bloque pendant dual outage : "
    f"{dual_star_tracker_blocked}"
)


print()


print(
    f"VALIDATION GLOBALE : "
    f"{logic_validated}"
)


print(
    "=============================================================================================================="
)


# ------------------------------------------------------------
# 9. CONVERSION MODE -> INDICE POUR FIGURE
# ------------------------------------------------------------

mode_to_index = {
    ResilienceMode.NOMINAL:
        0,

    ResilienceMode.RECOVERY:
        1,

    ResilienceMode.GNSS_RECONFIGURED:
        2,

    ResilienceMode.ATTITUDE_DEGRADED:
        3,

    ResilienceMode.GNSS_OUTAGE:
        4,

    ResilienceMode.GNSS_REJECTED:
        5,

    ResilienceMode.DUAL_OUTAGE:
        6
}


mode_numeric_history = np.array([
    mode_to_index[
        mode
    ]
    for mode in mode_history
])


# ------------------------------------------------------------
# 10. FIGURE MODES
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)


plt.step(
    time_minutes,
    mode_numeric_history,
    where="post"
)


plt.yticks(
    list(
        mode_to_index.values()
    ),

    [
        mode.value
        for mode in mode_to_index.keys()
    ]
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Mode AURORA"
)


plt.title(
    "AURORA — Machine d'etats de resilience"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 11. FIGURE SEVERITE
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)


plt.step(
    time_minutes,
    severity_history,
    where="post"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Niveau de severite"
)


plt.yticks([
    0,
    1,
    2,
    3
])


plt.title(
    "AURORA — Niveau de degradation operationnelle"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# 12. FIGURE ACTIONS CAPTEURS
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 5)
)


plt.step(
    time_minutes,
    gnss_used_history.astype(
        int
    ),
    where="post",
    label="Update GNSS autorise"
)


plt.step(
    time_minutes,
    star_tracker_used_history.astype(
        int
    ),
    where="post",
    label="Update star tracker autorise"
)


plt.xlabel(
    "Temps [min]"
)


plt.ylabel(
    "Autorisation"
)


plt.yticks([
    0,
    1
])


plt.title(
    "AURORA — Reconfiguration des mises a jour capteurs"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()