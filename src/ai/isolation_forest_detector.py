import numpy as np

from sklearn.ensemble import IsolationForest


# ============================================================
# AURORA
# Isolation Forest anomaly detector
#
# Phase 15-B
#
# Philosophy:
#
#   Train ONLY on nominal GNSS data.
#
#   Higher anomaly_score
#       =>
#   more abnormal according to learned nominal distribution.
#
#
# IMPORTANT:
#
# Threshold calibration is performed separately on an
# independent NOMINAL calibration set.
#
# Test data must never be used to:
#
#   - train the model
#   - calibrate the threshold
# ============================================================


class IsolationForestAnomalyDetector:

    def __init__(
        self,
        feature_columns,
        n_estimators=400,
        random_state=1502
    ):

        self.feature_columns = list(
            feature_columns
        )

        self.n_estimators = int(
            n_estimators
        )

        self.random_state = int(
            random_state
        )

        self.model = IsolationForest(
            n_estimators=
                self.n_estimators,

            max_samples=
                "auto",

            contamination=
                "auto",

            max_features=
                1.0,

            bootstrap=
                False,

            random_state=
                self.random_state,

            n_jobs=
                -1
        )

        self.threshold = (
            None
        )

        self.target_false_alarm_rate = (
            None
        )

        self.is_fitted = (
            False
        )


    # --------------------------------------------------------
    # DATA CONVERSION
    # --------------------------------------------------------

    def dataframe_to_matrix(
        self,
        dataframe
    ):

        missing_columns = [
            column
            for column in self.feature_columns
            if column
            not in dataframe.columns
        ]

        if len(
            missing_columns
        ) > 0:

            raise ValueError(
                "Features manquantes : "
                +
                str(
                    missing_columns
                )
            )


        matrix = dataframe[
            self.feature_columns
        ].to_numpy(
            dtype=float
        )


        if not np.all(
            np.isfinite(
                matrix
            )
        ):

            raise ValueError(
                "Le vecteur de features contient "
                "des valeurs non finies."
            )


        return matrix


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit_nominal(
        self,
        nominal_dataframe
    ):

        if len(
            nominal_dataframe
        ) == 0:

            raise ValueError(
                "Le dataset nominal d'entrainement est vide."
            )


        feature_matrix = (
            self.dataframe_to_matrix(
                nominal_dataframe
            )
        )


        self.model.fit(
            feature_matrix
        )


        self.is_fitted = (
            True
        )

        self.threshold = (
            None
        )


        return self


    # --------------------------------------------------------
    # ANOMALY SCORE
    # --------------------------------------------------------

    def score_samples(
        self,
        dataframe
    ):
        """
        Returns an anomaly score where HIGHER means
        MORE anomalous.

        sklearn IsolationForest score_samples:
            higher = more normal

        therefore:
            anomaly_score = -score_samples
        """

        if not self.is_fitted:

            raise RuntimeError(
                "Le detecteur doit etre entraine avant scoring."
            )


        feature_matrix = (
            self.dataframe_to_matrix(
                dataframe
            )
        )


        normality_score = (
            self.model.score_samples(
                feature_matrix
            )
        )


        anomaly_score = (
            -normality_score
        )


        return anomaly_score


    # --------------------------------------------------------
    # THRESHOLD CALIBRATION
    # --------------------------------------------------------

    def calibrate_threshold(
        self,
        nominal_calibration_dataframe,
        target_false_alarm_rate=0.01
    ):
        """
        Calibrate threshold only from nominal data.

        For N calibration samples and target alpha:

            K = ceil(alpha * N)

        threshold is selected so approximately K nominal
        samples are declared anomalous.
        """

        if not self.is_fitted:

            raise RuntimeError(
                "Entrainer le detecteur avant calibration."
            )


        target_false_alarm_rate = float(
            target_false_alarm_rate
        )


        if not (
            0.0
            <
            target_false_alarm_rate
            <
            1.0
        ):

            raise ValueError(
                "target_false_alarm_rate doit etre "
                "strictement entre 0 et 1."
            )


        scores = (
            self.score_samples(
                nominal_calibration_dataframe
            )
        )


        number_of_samples = len(
            scores
        )


        if number_of_samples < 2:

            raise ValueError(
                "Pas assez d'echantillons de calibration."
            )


        number_allowed_false_alarms = max(
            1,
            int(
                np.ceil(
                    target_false_alarm_rate
                    *
                    number_of_samples
                )
            )
        )


        sorted_scores = np.sort(
            scores
        )


        threshold_index = (
            number_of_samples
            -
            number_allowed_false_alarms
        )


        threshold_index = np.clip(
            threshold_index,
            0,
            number_of_samples - 1
        )


        self.threshold = float(
            sorted_scores[
                threshold_index
            ]
        )


        self.target_false_alarm_rate = (
            target_false_alarm_rate
        )


        calibration_predictions = (
            scores
            >=
            self.threshold
        )


        achieved_false_alarm_rate = (
            np.mean(
                calibration_predictions
            )
        )


        return {
            "threshold":
                self.threshold,

            "target_false_alarm_rate":
                target_false_alarm_rate,

            "achieved_false_alarm_rate":
                achieved_false_alarm_rate,

            "number_of_samples":
                number_of_samples,

            "number_allowed_false_alarms":
                number_allowed_false_alarms
        }


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        dataframe
    ):

        if self.threshold is None:

            raise RuntimeError(
                "Le seuil doit etre calibre avant prediction."
            )


        scores = (
            self.score_samples(
                dataframe
            )
        )


        predictions = (
            scores
            >=
            self.threshold
        )


        return predictions.astype(
            int
        )