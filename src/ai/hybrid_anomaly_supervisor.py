from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


# ============================================================
# AURORA
# Phase 15-D
#
# Persistence-aware hybrid GNSS anomaly supervisor
#
#
# Principle:
#
# Classical FDIR alarm
#       ->
# immediate hard alarm
#
#
# AI anomaly
#       ->
# advisory evidence first
#
# repeated AI evidence
#       ->
# confirmed AI anomaly
#       ->
# hard alarm
#
#
# This prevents a single isolated ML classification from
# immediately becoming a flight-critical GNSS rejection.
#
#
# Everything is causal:
#
# only present and past AI decisions are stored.
# ============================================================


class HybridAnomalyState(
    str,
    Enum
):

    CLEAR = (
        "CLEAR"
    )

    AI_WATCH = (
        "AI_WATCH"
    )

    AI_CONFIRMED = (
        "AI_CONFIRMED"
    )

    CLASSICAL_ALARM = (
        "CLASSICAL_ALARM"
    )


@dataclass
class HybridAnomalyDecision:

    state: HybridAnomalyState

    hard_alarm: bool

    classical_alarm: bool

    ai_raw_alarm: bool

    ai_confirmed: bool

    ai_vote_count: int

    ai_history_length: int

    reason: str


class HybridGnssAnomalySupervisor:
    """
    Causal evidence-fusion supervisor.

    AI confirmation rule:

        at least minimum_ai_votes anomalies
        inside the last ai_vote_window samples.

    Example:

        window = 3
        minimum votes = 2

        [0, 1, 1]
            -> AI confirmed

        [0, 0, 1]
            -> AI watch only

    Classical FDIR always has immediate priority.
    """

    def __init__(
        self,
        ai_vote_window=3,
        minimum_ai_votes=2
    ):

        self.ai_vote_window = int(
            ai_vote_window
        )

        self.minimum_ai_votes = int(
            minimum_ai_votes
        )


        if self.ai_vote_window < 1:

            raise ValueError(
                "ai_vote_window doit etre >= 1."
            )


        if self.minimum_ai_votes < 1:

            raise ValueError(
                "minimum_ai_votes doit etre >= 1."
            )


        if (
            self.minimum_ai_votes
            >
            self.ai_vote_window
        ):

            raise ValueError(
                "minimum_ai_votes ne peut pas "
                "depasser ai_vote_window."
            )


        self.ai_history = deque(
            maxlen=
                self.ai_vote_window
        )


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(
        self
    ):

        self.ai_history.clear()


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        classical_alarm,
        ai_raw_alarm
    ):

        classical_alarm = bool(
            classical_alarm
        )

        ai_raw_alarm = bool(
            ai_raw_alarm
        )


        # ====================================================
        # Store causal AI evidence
        # ====================================================

        self.ai_history.append(
            int(
                ai_raw_alarm
            )
        )


        ai_vote_count = int(
            np.sum(
                self.ai_history
            )
        )


        ai_history_length = len(
            self.ai_history
        )


        # ====================================================
        # We require enough actual observations to collect
        # the requested number of votes.
        # ====================================================

        ai_confirmed = (
            ai_history_length
            >=
            self.minimum_ai_votes
            and
            ai_vote_count
            >=
            self.minimum_ai_votes
        )


        # ====================================================
        # Classical FDIR has immediate priority.
        # ====================================================

        if classical_alarm:

            return HybridAnomalyDecision(
                state=
                    HybridAnomalyState.CLASSICAL_ALARM,

                hard_alarm=
                    True,

                classical_alarm=
                    True,

                ai_raw_alarm=
                    ai_raw_alarm,

                ai_confirmed=
                    ai_confirmed,

                ai_vote_count=
                    ai_vote_count,

                ai_history_length=
                    ai_history_length,

                reason=
                    (
                        "Alarme FDIR classique : "
                        "alarme dure immediate."
                    )
            )


        # ====================================================
        # Persistent AI evidence
        # ====================================================

        if ai_confirmed:

            return HybridAnomalyDecision(
                state=
                    HybridAnomalyState.AI_CONFIRMED,

                hard_alarm=
                    True,

                classical_alarm=
                    False,

                ai_raw_alarm=
                    ai_raw_alarm,

                ai_confirmed=
                    True,

                ai_vote_count=
                    ai_vote_count,

                ai_history_length=
                    ai_history_length,

                reason=
                    (
                        "Anomalie AI confirmee par "
                        "persistance temporelle."
                    )
            )


        # ====================================================
        # Single / insufficient AI evidence
        # ====================================================

        if ai_raw_alarm:

            return HybridAnomalyDecision(
                state=
                    HybridAnomalyState.AI_WATCH,

                hard_alarm=
                    False,

                classical_alarm=
                    False,

                ai_raw_alarm=
                    True,

                ai_confirmed=
                    False,

                ai_vote_count=
                    ai_vote_count,

                ai_history_length=
                    ai_history_length,

                reason=
                    (
                        "Evidence AI presente mais "
                        "persistance insuffisante."
                    )
            )


        # ====================================================
        # Clear
        # ====================================================

        return HybridAnomalyDecision(
            state=
                HybridAnomalyState.CLEAR,

            hard_alarm=
                False,

            classical_alarm=
                False,

            ai_raw_alarm=
                False,

            ai_confirmed=
                False,

            ai_vote_count=
                ai_vote_count,

            ai_history_length=
                ai_history_length,

            reason=
                "Aucune anomalie confirmee."
        )