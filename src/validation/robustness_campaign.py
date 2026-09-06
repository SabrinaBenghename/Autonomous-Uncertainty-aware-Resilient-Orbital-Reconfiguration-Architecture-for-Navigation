from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.stats import qmc


# ============================================================
# AURORA
# Phase 16
#
# Robustness campaign design
#
#
# Purpose:
#
# Generate a deterministic, reproducible and space-filling
# set of robustness scenarios using Latin Hypercube Sampling.
#
#
# IMPORTANT:
#
# The parameter ranges below are ENGINEERING STRESS-TEST
# ranges for the AURORA simulation.
#
# They are NOT claimed to be flight-qualified probability
# distributions for a specific CubeSat sensor suite.
#
#
# Orbit altitude / inclination are deliberately excluded.
#
# Orbital scenario variation belongs to Phase 17.
# ============================================================


# ------------------------------------------------------------
# 1. PARAMETER SPECIFICATION
# ------------------------------------------------------------

@dataclass(
    frozen=True
)
class RobustnessParameterSpec:

    name: str

    lower: float

    upper: float

    scaling: str

    unit: str

    description: str


    def __post_init__(
        self
    ):

        if not np.isfinite(
            self.lower
        ):

            raise ValueError(
                f"Lower bound non finite for {self.name}."
            )


        if not np.isfinite(
            self.upper
        ):

            raise ValueError(
                f"Upper bound non finite for {self.name}."
            )


        if (
            self.upper
            <=
            self.lower
        ):

            raise ValueError(
                f"Invalid bounds for {self.name}: "
                f"[{self.lower}, {self.upper}]"
            )


        if self.scaling not in [
            "linear",
            "log"
        ]:

            raise ValueError(
                f"Unknown scaling for {self.name}: "
                f"{self.scaling}"
            )


        if (
            self.scaling
            ==
            "log"
            and
            self.lower
            <=
            0.0
        ):

            raise ValueError(
                f"Log-scaled parameter {self.name} "
                "must have positive bounds."
            )


    # --------------------------------------------------------
    # UNIT INTERVAL -> PHYSICAL VALUE
    # --------------------------------------------------------

    def transform(
        self,
        unit_values
    ):

        unit_values = np.asarray(
            unit_values,
            dtype=float
        )


        if np.any(
            unit_values
            <
            0.0
        ):

            raise ValueError(
                f"Unit samples below zero for {self.name}."
            )


        if np.any(
            unit_values
            >
            1.0
        ):

            raise ValueError(
                f"Unit samples above one for {self.name}."
            )


        if self.scaling == "linear":

            return (
                self.lower
                +
                unit_values
                *
                (
                    self.upper
                    -
                    self.lower
                )
            )


        log_lower = np.log(
            self.lower
        )


        log_upper = np.log(
            self.upper
        )


        return np.exp(
            log_lower
            +
            unit_values
            *
            (
                log_upper
                -
                log_lower
            )
        )


    # --------------------------------------------------------
    # PHYSICAL VALUE -> UNIT INTERVAL
    # --------------------------------------------------------

    def inverse_transform(
        self,
        physical_values
    ):

        physical_values = np.asarray(
            physical_values,
            dtype=float
        )


        if self.scaling == "linear":

            return (
                physical_values
                -
                self.lower
            ) / (
                self.upper
                -
                self.lower
            )


        return (
            np.log(
                physical_values
            )
            -
            np.log(
                self.lower
            )
        ) / (
            np.log(
                self.upper
            )
            -
            np.log(
                self.lower
            )
        )


# ------------------------------------------------------------
# 2. PHASE-16 DEFAULT PARAMETER SPACE
# ------------------------------------------------------------

def build_phase16_parameter_specs():
    """
    Engineering robustness ranges.

    These ranges deliberately surround the nominal values
    used in Phases 8-15.
    """

    return [

        # ====================================================
        # GNSS
        # ====================================================

        RobustnessParameterSpec(
            name=
                "pseudorange_noise_std_m",

            lower=
                1.5,

            upper=
                6.0,

            scaling=
                "log",

            unit=
                "m",

            description=
                "GNSS pseudorange measurement noise std."
        ),


        # ====================================================
        # ACCELEROMETER
        # ====================================================

        RobustnessParameterSpec(
            name=
                "accelerometer_noise_std_mps2",

            lower=
                2.5e-7,

            upper=
                1.0e-6,

            scaling=
                "log",

            unit=
                "m/s^2",

            description=
                "Body-frame accelerometer white-noise std."
        ),


        RobustnessParameterSpec(
            name=
                "accelerometer_bias_x_mps2",

            lower=
                -5.0e-6,

            upper=
                5.0e-6,

            scaling=
                "linear",

            unit=
                "m/s^2",

            description=
                "True accelerometer X-body bias."
        ),


        RobustnessParameterSpec(
            name=
                "accelerometer_bias_y_mps2",

            lower=
                -5.0e-6,

            upper=
                5.0e-6,

            scaling=
                "linear",

            unit=
                "m/s^2",

            description=
                "True accelerometer Y-body bias."
        ),


        RobustnessParameterSpec(
            name=
                "accelerometer_bias_z_mps2",

            lower=
                -5.0e-6,

            upper=
                5.0e-6,

            scaling=
                "linear",

            unit=
                "m/s^2",

            description=
                "True accelerometer Z-body bias."
        ),


        # ====================================================
        # GYROSCOPE
        # ====================================================

        RobustnessParameterSpec(
            name=
                "gyro_noise_std_degps",

            lower=
                2.5e-4,

            upper=
                1.0e-3,

            scaling=
                "log",

            unit=
                "deg/s",

            description=
                "Gyroscope white-noise std."
        ),


        RobustnessParameterSpec(
            name=
                "gyro_bias_x_degps",

            lower=
                -0.002,

            upper=
                0.002,

            scaling=
                "linear",

            unit=
                "deg/s",

            description=
                "True gyroscope X-body bias."
        ),


        RobustnessParameterSpec(
            name=
                "gyro_bias_y_degps",

            lower=
                -0.002,

            upper=
                0.002,

            scaling=
                "linear",

            unit=
                "deg/s",

            description=
                "True gyroscope Y-body bias."
        ),


        RobustnessParameterSpec(
            name=
                "gyro_bias_z_degps",

            lower=
                -0.002,

            upper=
                0.002,

            scaling=
                "linear",

            unit=
                "deg/s",

            description=
                "True gyroscope Z-body bias."
        ),


        # ====================================================
        # STAR TRACKER
        # ====================================================

        RobustnessParameterSpec(
            name=
                "star_tracker_noise_std_deg",

            lower=
                0.01,

            upper=
                0.05,

            scaling=
                "log",

            unit=
                "deg",

            description=
                "Absolute star-tracker attitude noise std."
        ),


        # ====================================================
        # INITIAL POSITION
        # ====================================================

        RobustnessParameterSpec(
            name=
                "initial_position_error_x_m",

            lower=
                -200.0,

            upper=
                200.0,

            scaling=
                "linear",

            unit=
                "m",

            description=
                "Initial navigation position X error."
        ),


        RobustnessParameterSpec(
            name=
                "initial_position_error_y_m",

            lower=
                -200.0,

            upper=
                200.0,

            scaling=
                "linear",

            unit=
                "m",

            description=
                "Initial navigation position Y error."
        ),


        RobustnessParameterSpec(
            name=
                "initial_position_error_z_m",

            lower=
                -200.0,

            upper=
                200.0,

            scaling=
                "linear",

            unit=
                "m",

            description=
                "Initial navigation position Z error."
        ),


        # ====================================================
        # INITIAL VELOCITY
        # ====================================================

        RobustnessParameterSpec(
            name=
                "initial_velocity_error_x_mps",

            lower=
                -0.20,

            upper=
                0.20,

            scaling=
                "linear",

            unit=
                "m/s",

            description=
                "Initial navigation velocity X error."
        ),


        RobustnessParameterSpec(
            name=
                "initial_velocity_error_y_mps",

            lower=
                -0.20,

            upper=
                0.20,

            scaling=
                "linear",

            unit=
                "m/s",

            description=
                "Initial navigation velocity Y error."
        ),


        RobustnessParameterSpec(
            name=
                "initial_velocity_error_z_mps",

            lower=
                -0.20,

            upper=
                0.20,

            scaling=
                "linear",

            unit=
                "m/s",

            description=
                "Initial navigation velocity Z error."
        ),


        # ====================================================
        # NON-GRAVITATIONAL FORCE
        # ====================================================

        RobustnessParameterSpec(
            name=
                "non_gravitational_acceleration_mps2",

            lower=
                1.0e-5,

            upper=
                4.0e-5,

            scaling=
                "log",

            unit=
                "m/s^2",

            description=
                "Base synthetic drag-like acceleration."
        ),


        # ====================================================
        # LOW-REDUNDANCY GNSS FAULT
        # ====================================================

        RobustnessParameterSpec(
            name=
                "fault_magnitude_m",

            lower=
                5.0,

            upper=
                80.0,

            scaling=
                "linear",

            unit=
                "m",

            description=
                "Persistent single-satellite pseudorange fault."
        ),


        RobustnessParameterSpec(
            name=
                "fault_start_min",

            lower=
                100.0,

            upper=
                110.0,

            scaling=
                "linear",

            unit=
                "min",

            description=
                "Start time of low-redundancy GNSS fault."
        ),


        RobustnessParameterSpec(
            name=
                "fault_duration_min",

            lower=
                5.0,

            upper=
                10.0,

            scaling=
                "linear",

            unit=
                "min",

            description=
                "Duration of low-redundancy GNSS fault."
        ),


        # ====================================================
        # OUTAGES
        # ====================================================

        RobustnessParameterSpec(
            name=
                "gnss_outage_duration_min",

            lower=
                10.0,

            upper=
                20.0,

            scaling=
                "linear",

            unit=
                "min",

            description=
                "Duration of early complete GNSS outage."
        ),


        RobustnessParameterSpec(
            name=
                "star_tracker_outage_duration_min",

            lower=
                5.0,

            upper=
                15.0,

            scaling=
                "linear",

            unit=
                "min",

            description=
                "Duration of star-tracker-only outage."
        ),


        RobustnessParameterSpec(
            name=
                "dual_outage_duration_min",

            lower=
                10.0,

            upper=
                20.0,

            scaling=
                "linear",

            unit=
                "min",

            description=
                "Duration of simultaneous GNSS/ST outage."
        )
    ]


# ------------------------------------------------------------
# 3. CAMPAIGN CONTAINER
# ------------------------------------------------------------

@dataclass
class RobustnessCampaign:

    dataframe: pd.DataFrame

    unit_samples: np.ndarray

    parameter_specs: list

    seed: int


# ------------------------------------------------------------
# 4. GENERATE LATIN HYPERCUBE CAMPAIGN
# ------------------------------------------------------------

def generate_latin_hypercube_campaign(
    number_of_runs=64,
    seed=16001,
    parameter_specs=None
):

    number_of_runs = int(
        number_of_runs
    )


    seed = int(
        seed
    )


    if number_of_runs < 2:

        raise ValueError(
            "number_of_runs doit etre >= 2."
        )


    if parameter_specs is None:

        parameter_specs = (
            build_phase16_parameter_specs()
        )


    number_of_dimensions = len(
        parameter_specs
    )


    if number_of_dimensions == 0:

        raise ValueError(
            "Aucun parametre de campagne."
        )


    # ========================================================
    # Latin Hypercube in unit cube [0,1]^D
    # ========================================================

    sampler = qmc.LatinHypercube(
        d=
            number_of_dimensions,

        seed=
            seed
    )


    unit_samples = sampler.random(
        n=
            number_of_runs
    )


    # ========================================================
    # Physical parameter mapping
    # ========================================================

    data = {}


    for dimension_index, spec in enumerate(
        parameter_specs
    ):

        data[
            spec.name
        ] = spec.transform(
            unit_samples[
                :,
                dimension_index
            ]
        )


    dataframe = pd.DataFrame(
        data
    )


    # ========================================================
    # Deterministic IDs / seeds
    # ========================================================

    dataframe.insert(
        0,
        "scenario_id",
        [
            f"AURORA16_{index:03d}"
            for index in range(
                number_of_runs
            )
        ]
    )


    dataframe.insert(
        1,
        "scenario_index",
        np.arange(
            number_of_runs,
            dtype=int
        )
    )


    base_scenario_seed = (
        1_600_000
        +
        1000
        *
        (
            seed
            %
            1000
        )
    )


    dataframe.insert(
        2,
        "scenario_seed",
        base_scenario_seed
        +
        np.arange(
            number_of_runs,
            dtype=int
        )
    )


    return RobustnessCampaign(
        dataframe=
            dataframe,

        unit_samples=
            unit_samples,

        parameter_specs=
            list(
                parameter_specs
            ),

        seed=
            seed
    )


# ------------------------------------------------------------
# 5. PARAMETER SUMMARY
# ------------------------------------------------------------

def summarize_campaign(
    campaign
):

    rows = []


    dataframe = (
        campaign.dataframe
    )


    for spec in campaign.parameter_specs:

        values = dataframe[
            spec.name
        ].to_numpy(
            dtype=float
        )


        rows.append(
            {
                "parameter":
                    spec.name,

                "unit":
                    spec.unit,

                "scaling":
                    spec.scaling,

                "design_lower":
                    spec.lower,

                "sample_min":
                    np.min(
                        values
                    ),

                "sample_mean":
                    np.mean(
                        values
                    ),

                "sample_std":
                    np.std(
                        values,
                        ddof=1
                    ),

                "sample_max":
                    np.max(
                        values
                    ),

                "design_upper":
                    spec.upper,

                "description":
                    spec.description
            }
        )


    return pd.DataFrame(
        rows
    )


# ------------------------------------------------------------
# 6. BOUNDS VALIDATION
# ------------------------------------------------------------

def campaign_values_within_bounds(
    campaign,
    tolerance=1.0e-12
):

    for spec in campaign.parameter_specs:

        values = (
            campaign.dataframe[
                spec.name
            ].to_numpy(
                dtype=float
            )
        )


        if np.any(
            values
            <
            spec.lower
            -
            tolerance
        ):

            return False


        if np.any(
            values
            >
            spec.upper
            +
            tolerance
        ):

            return False


    return True


# ------------------------------------------------------------
# 7. LATIN-HYPERCUBE STRATIFICATION VALIDATION
# ------------------------------------------------------------

def latin_hypercube_stratification_valid(
    campaign
):
    """
    For N samples, a Latin Hypercube must contain exactly
    one sample in each of the N 1-D strata for every
    parameter dimension.
    """

    unit_samples = np.asarray(
        campaign.unit_samples,
        dtype=float
    )


    number_of_runs = (
        unit_samples.shape[
            0
        ]
    )


    expected_strata = np.arange(
        number_of_runs,
        dtype=int
    )


    for dimension_index in range(
        unit_samples.shape[
            1
        ]
    ):

        strata = np.floor(
            unit_samples[
                :,
                dimension_index
            ]
            *
            number_of_runs
        ).astype(
            int
        )


        strata = np.clip(
            strata,
            0,
            number_of_runs - 1
        )


        if not np.array_equal(
            np.sort(
                strata
            ),
            expected_strata
        ):

            return False


    return True


# ------------------------------------------------------------
# 8. FINITE VALIDATION
# ------------------------------------------------------------

def campaign_values_finite(
    campaign
):

    parameter_names = [
        spec.name
        for spec in campaign.parameter_specs
    ]


    matrix = (
        campaign.dataframe[
            parameter_names
        ].to_numpy(
            dtype=float
        )
    )


    return bool(
        np.all(
            np.isfinite(
                matrix
            )
        )
    )


# ------------------------------------------------------------
# 9. REPRODUCIBILITY VALIDATION
# ------------------------------------------------------------

def campaigns_identical(
    campaign_a,
    campaign_b
):

    if (
        campaign_a.seed
        !=
        campaign_b.seed
    ):

        return False


    if (
        campaign_a.dataframe.shape
        !=
        campaign_b.dataframe.shape
    ):

        return False


    if not np.array_equal(
        campaign_a.unit_samples,
        campaign_b.unit_samples
    ):

        return False


    return campaign_a.dataframe.equals(
        campaign_b.dataframe
    )


# ------------------------------------------------------------
# 10. TIMELINE CONSTRAINT VALIDATION
# ------------------------------------------------------------

def phase16_timeline_constraints_valid(
    campaign
):
    """
    Phase-16 nominal event anchors:

        early GNSS outage:
            starts at 40 min

        star-tracker outage:
            starts at 65 min

        low-redundancy fault:
            variable start

        dual outage:
            starts at 125 min

        mission end:
            150 min

    We ensure sampled durations do not create unintended
    overlaps outside the intended campaign design.
    """

    dataframe = campaign.dataframe


    early_gnss_outage_end = (
        40.0
        +
        dataframe[
            "gnss_outage_duration_min"
        ].to_numpy()
    )


    star_tracker_outage_end = (
        65.0
        +
        dataframe[
            "star_tracker_outage_duration_min"
        ].to_numpy()
    )


    low_redundancy_fault_end = (
        dataframe[
            "fault_start_min"
        ].to_numpy()
        +
        dataframe[
            "fault_duration_min"
        ].to_numpy()
    )


    dual_outage_end = (
        125.0
        +
        dataframe[
            "dual_outage_duration_min"
        ].to_numpy()
    )


    return bool(
        np.all(
            early_gnss_outage_end
            <=
            60.0
        )
        and
        np.all(
            star_tracker_outage_end
            <=
            80.0
        )
        and
        np.all(
            low_redundancy_fault_end
            <=
            120.0
        )
        and
        np.all(
            dual_outage_end
            <=
            145.0
        )
    )