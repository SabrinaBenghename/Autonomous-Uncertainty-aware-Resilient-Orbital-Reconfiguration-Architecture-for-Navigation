from dataclasses import dataclass
from enum import Enum


# ============================================================
# AURORA
# Resilience / navigation mode manager
#
# Distinction importante :
#
#   GNSS signal availability
#       !=
#   GNSS integrity availability
#
#
# Pour l'architecture FDIR actuelle :
#
#   - 4 satellites permettent une solution GNSS
#   - 5 satellites donnent une faible redondance
#   - >= 6 satellites permettent une isolation
#     leave-one-out avec redondance residuelle
#
#
# Le manager peut donc interdire un update GNSS meme si
# une position GNSS mathematique est disponible.
# ============================================================


# ------------------------------------------------------------
# 1. MODES
# ------------------------------------------------------------

class ResilienceMode(
    str,
    Enum
):
    NOMINAL = (
        "NOMINAL"
    )

    GNSS_RECONFIGURED = (
        "GNSS_RECONFIGURED"
    )

    GNSS_OUTAGE = (
        "GNSS_OUTAGE"
    )

    GNSS_REJECTED = (
        "GNSS_REJECTED"
    )

    GNSS_INTEGRITY_UNAVAILABLE = (
        "GNSS_INTEGRITY_UNAVAILABLE"
    )

    ATTITUDE_DEGRADED = (
        "ATTITUDE_DEGRADED"
    )

    DUAL_OUTAGE = (
        "DUAL_OUTAGE"
    )

    RECOVERY = (
        "RECOVERY"
    )


# ------------------------------------------------------------
# 2. DECISION
# ------------------------------------------------------------

@dataclass
class ResilienceDecision:

    mode: ResilienceMode

    raw_mode: ResilienceMode

    use_gnss_update: bool

    use_star_tracker_update: bool

    navigation_coasting: bool

    attitude_coasting: bool

    integrity_degraded: bool

    severity: int

    reason: str

    gnss_integrity_available: bool


# ------------------------------------------------------------
# 3. MANAGER
# ------------------------------------------------------------

class ResilienceManager:

    def __init__(
        self,
        attitude_sigma_limit_rad,
        recovery_epochs=6
    ):

        self.attitude_sigma_limit_rad = float(
            attitude_sigma_limit_rad
        )

        self.recovery_epochs = int(
            recovery_epochs
        )

        if (
            self.attitude_sigma_limit_rad
            <= 0.0
        ):

            raise ValueError(
                "La limite sigma attitude doit etre positive."
            )

        if (
            self.recovery_epochs
            < 0
        ):

            raise ValueError(
                "recovery_epochs doit etre >= 0."
            )

        self.reset()


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(
        self
    ):

        self.current_mode = (
            ResilienceMode.NOMINAL
        )

        self.recovery_remaining = (
            0
        )

        self.history = []


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    @staticmethod
    def mode_severity(
        mode
    ):

        severity_map = {
            ResilienceMode.NOMINAL:
                0,

            ResilienceMode.RECOVERY:
                1,

            ResilienceMode.GNSS_RECONFIGURED:
                1,

            ResilienceMode.GNSS_OUTAGE:
                2,

            ResilienceMode.GNSS_REJECTED:
                2,

            ResilienceMode.GNSS_INTEGRITY_UNAVAILABLE:
                2,

            ResilienceMode.ATTITUDE_DEGRADED:
                2,

            ResilienceMode.DUAL_OUTAGE:
                3
        }

        return severity_map[
            mode
        ]


    # --------------------------------------------------------
    # RAW MODE
    # --------------------------------------------------------

    def determine_raw_mode(
        self,
        gnss_signal_available,
        gnss_measurement_accepted,
        gnss_fault_detected,
        gnss_reconfigured,
        star_tracker_available,
        attitude_sigma_rad,
        gnss_integrity_available=True
    ):

        attitude_sigma_rad = float(
            attitude_sigma_rad
        )

        gnss_signal_available = bool(
            gnss_signal_available
        )

        gnss_measurement_accepted = bool(
            gnss_measurement_accepted
        )

        gnss_fault_detected = bool(
            gnss_fault_detected
        )

        gnss_reconfigured = bool(
            gnss_reconfigured
        )

        star_tracker_available = bool(
            star_tracker_available
        )

        gnss_integrity_available = bool(
            gnss_integrity_available
        )


        # ----------------------------------------------------
        # ATTITUDE HEALTH
        # ----------------------------------------------------

        attitude_sigma_exceeded = (
            attitude_sigma_rad
            >
            self.attitude_sigma_limit_rad
        )

        attitude_degraded = (
            (
                not star_tracker_available
            )
            or
            attitude_sigma_exceeded
        )


        # ----------------------------------------------------
        # GNSS EFFECTIVELY USABLE
        #
        # Signal alone is not enough.
        #
        # The measurement must:
        #
        #   - exist
        #   - be accepted by FDIR
        #   - have sufficient integrity capability
        # ----------------------------------------------------

        effective_gnss_available = (
            gnss_signal_available
            and
            gnss_measurement_accepted
            and
            gnss_integrity_available
        )


        # ----------------------------------------------------
        # 1. DOUBLE DEGRADATION
        # ----------------------------------------------------

        if (
            not effective_gnss_available
            and
            attitude_degraded
        ):

            return (
                ResilienceMode.DUAL_OUTAGE,
                effective_gnss_available,
                attitude_degraded,
                (
                    "Navigation absolue GNSS non exploitable "
                    "et attitude non aidee ou incertaine."
                )
            )


        # ----------------------------------------------------
        # 2. FAULT DETECTED AND REJECTED
        #
        # Keep this before the integrity-availability mode.
        #
        # If a fault is explicitly detected, GNSS_REJECTED
        # is the most informative diagnosis.
        # ----------------------------------------------------

        if (
            gnss_signal_available
            and
            gnss_fault_detected
            and
            not gnss_measurement_accepted
        ):

            return (
                ResilienceMode.GNSS_REJECTED,
                effective_gnss_available,
                attitude_degraded,
                (
                    "Faute GNSS detectee mais aucune "
                    "solution reconfiguree sure n'est disponible."
                )
            )


        # ----------------------------------------------------
        # 3. INTEGRITY UNAVAILABLE
        #
        # GNSS position may mathematically exist,
        # but AURORA does not consider it protected.
        # ----------------------------------------------------

        if (
            gnss_signal_available
            and
            not gnss_integrity_available
        ):

            return (
                ResilienceMode.GNSS_INTEGRITY_UNAVAILABLE,
                effective_gnss_available,
                attitude_degraded,
                (
                    "Signal GNSS disponible mais redondance "
                    "insuffisante pour garantir la fonction "
                    "FDIR/isolation requise."
                )
            )


        # ----------------------------------------------------
        # 4. GNSS PHYSICAL / COMPUTATIONAL OUTAGE
        # ----------------------------------------------------

        if (
            not effective_gnss_available
        ):

            return (
                ResilienceMode.GNSS_OUTAGE,
                effective_gnss_available,
                attitude_degraded,
                "Aucune mesure GNSS exploitable."
            )


        # ----------------------------------------------------
        # 5. ATTITUDE DEGRADEE
        # ----------------------------------------------------

        if attitude_degraded:

            if (
                not star_tracker_available
            ):

                reason = (
                    "Star tracker indisponible : "
                    "attitude propagee par gyro/MEKF."
                )

            else:

                reason = (
                    "Incertitude attitude superieure "
                    "a la limite operationnelle."
                )

            return (
                ResilienceMode.ATTITUDE_DEGRADED,
                effective_gnss_available,
                attitude_degraded,
                reason
            )


        # ----------------------------------------------------
        # 6. GNSS FAULT SUCCESSFULLY RECONFIGURED
        # ----------------------------------------------------

        if (
            gnss_fault_detected
            and
            gnss_reconfigured
        ):

            return (
                ResilienceMode.GNSS_RECONFIGURED,
                effective_gnss_available,
                attitude_degraded,
                (
                    "Faute GNSS detectee et satellite "
                    "fautif isole/rejete avec solution "
                    "GNSS reconfiguree valide."
                )
            )


        # ----------------------------------------------------
        # 7. NOMINAL
        # ----------------------------------------------------

        return (
            ResilienceMode.NOMINAL,
            effective_gnss_available,
            attitude_degraded,
            "Tous les sous-systemes sont nominaux."
        )


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        gnss_signal_available,
        gnss_measurement_accepted,
        gnss_fault_detected=False,
        gnss_reconfigured=False,
        star_tracker_available=True,
        attitude_sigma_rad=0.0,
        gnss_integrity_available=True
    ):

        (
            raw_mode,
            effective_gnss_available,
            attitude_degraded,
            raw_reason
        ) = self.determine_raw_mode(
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
                attitude_sigma_rad,

            gnss_integrity_available=
                gnss_integrity_available
        )


        previous_mode = (
            self.current_mode
        )


        # ----------------------------------------------------
        # ACTIVE DEGRADED MODE
        # ----------------------------------------------------

        if (
            raw_mode
            !=
            ResilienceMode.NOMINAL
        ):

            selected_mode = (
                raw_mode
            )

            self.recovery_remaining = (
                self.recovery_epochs
            )

            reason = (
                raw_reason
            )


        # ----------------------------------------------------
        # NOMINAL PHYSICS, POSSIBLE RECOVERY DELAY
        # ----------------------------------------------------

        else:

            if (
                previous_mode
                not in
                (
                    ResilienceMode.NOMINAL,
                    ResilienceMode.RECOVERY
                )
                and
                self.recovery_epochs
                >
                0
            ):

                self.recovery_remaining = (
                    self.recovery_epochs
                )


            if (
                self.recovery_remaining
                >
                0
            ):

                selected_mode = (
                    ResilienceMode.RECOVERY
                )

                reason = (
                    "Capteurs revenus nominaux ; "
                    "periode de stabilisation avant "
                    "retour au mode NOMINAL."
                )

                self.recovery_remaining -= (
                    1
                )

            else:

                selected_mode = (
                    ResilienceMode.NOMINAL
                )

                reason = (
                    raw_reason
                )


        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        use_gnss_update = (
            effective_gnss_available
        )

        use_star_tracker_update = (
            bool(
                star_tracker_available
            )
        )

        navigation_coasting = (
            not use_gnss_update
        )

        attitude_coasting = (
            not use_star_tracker_update
        )

        integrity_degraded = (
            selected_mode
            !=
            ResilienceMode.NOMINAL
        )

        severity = (
            self.mode_severity(
                selected_mode
            )
        )


        decision = (
            ResilienceDecision(
                mode=
                    selected_mode,

                raw_mode=
                    raw_mode,

                use_gnss_update=
                    use_gnss_update,

                use_star_tracker_update=
                    use_star_tracker_update,

                navigation_coasting=
                    navigation_coasting,

                attitude_coasting=
                    attitude_coasting,

                integrity_degraded=
                    integrity_degraded,

                severity=
                    severity,

                reason=
                    reason,

                gnss_integrity_available=
                    bool(
                        gnss_integrity_available
                    )
            )
        )


        self.current_mode = (
            selected_mode
        )

        self.history.append(
            decision
        )

        return decision