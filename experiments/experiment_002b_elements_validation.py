import numpy as np

from src.dynamics.orbit import R_EARTH

from src.dynamics.orbital_elements import (
    keplerian_to_cartesian,
    cartesian_to_keplerian
)


# ============================================================
# AURORA
# Expérience 002-B
# Validation des éléments orbitaux képlériens
# ============================================================


# ------------------------------------------------------------
# 1. ELEMENTS ORBITAUX DE REFERENCE
# ------------------------------------------------------------

reference_elements = {
    "semi_major_axis":
        R_EARTH + 550_000.0,

    "eccentricity":
        0.01,

    "inclination":
        np.deg2rad(97.6),

    "raan":
        np.deg2rad(40.0),

    "argument_of_periapsis":
        np.deg2rad(30.0),

    "true_anomaly":
        np.deg2rad(25.0)
}


# ------------------------------------------------------------
# 2. CONVERSION KEPLERIEN -> CARTESIEN
# ------------------------------------------------------------

position, velocity = (
    keplerian_to_cartesian(
        reference_elements[
            "semi_major_axis"
        ],
        reference_elements[
            "eccentricity"
        ],
        reference_elements[
            "inclination"
        ],
        reference_elements[
            "raan"
        ],
        reference_elements[
            "argument_of_periapsis"
        ],
        reference_elements[
            "true_anomaly"
        ]
    )
)


# ------------------------------------------------------------
# 3. CONVERSION CARTESIEN -> KEPLERIEN
# ------------------------------------------------------------

recovered_elements = (
    cartesian_to_keplerian(
        position,
        velocity
    )
)


# ------------------------------------------------------------
# 4. AFFICHAGE
# ------------------------------------------------------------

print(
    "\n"
    "=============================================="
)

print(
    "AURORA — Expérience 002-B"
)

print(
    "Validation des éléments képlériens"
)

print(
    "=============================================="
)


print(
    "\nDEMI-GRAND AXE"
)

print(
    f"Référence : "
    f"{reference_elements['semi_major_axis']/1000:.6f} km"
)

print(
    f"Retrouvé  : "
    f"{recovered_elements['semi_major_axis']/1000:.6f} km"
)


print(
    "\nEXCENTRICITÉ"
)

print(
    f"Référence : "
    f"{reference_elements['eccentricity']:.10f}"
)

print(
    f"Retrouvée : "
    f"{recovered_elements['eccentricity']:.10f}"
)


print(
    "\nINCLINAISON"
)

print(
    f"Référence : "
    f"{np.rad2deg(reference_elements['inclination']):.6f} deg"
)

print(
    f"Retrouvée : "
    f"{np.rad2deg(recovered_elements['inclination']):.6f} deg"
)


print(
    "\nRAAN"
)

print(
    f"Référence : "
    f"{np.rad2deg(reference_elements['raan']):.6f} deg"
)

print(
    f"Retrouvé  : "
    f"{np.rad2deg(recovered_elements['raan']):.6f} deg"
)


print(
    "\nARGUMENT DU PÉRIGÉE"
)

print(
    f"Référence : "
    f"{np.rad2deg(reference_elements['argument_of_periapsis']):.6f} deg"
)

print(
    f"Retrouvé  : "
    f"{np.rad2deg(recovered_elements['argument_of_periapsis']):.6f} deg"
)


print(
    "\nANOMALIE VRAIE"
)

print(
    f"Référence : "
    f"{np.rad2deg(reference_elements['true_anomaly']):.6f} deg"
)

print(
    f"Retrouvée : "
    f"{np.rad2deg(recovered_elements['true_anomaly']):.6f} deg"
)


print(
    "\n=============================================="
)