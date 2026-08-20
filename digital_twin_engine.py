###############################################################
# FOOD DIGITAL TWIN ENGINE
# CLOUD-READY VERSION 2.0
#
# Product-independent Digital Twin Engine
#
# Designed for:
#   Local PC
#   Render
#   Cloud Server
#   Future Mobile Application
#
# IMPORTANT:
#   No Windows-specific paths are used.
#   Data and models are loaded relative to the project folder.
###############################################################

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


###############################################################
# DIGITAL TWIN ENGINE
###############################################################

class DigitalTwinEngine:

    def __init__(self, product_config):

        self.config = product_config

        #######################################################
        # PROJECT ROOT
        #######################################################

        self.project_root = Path(
            __file__
        ).resolve().parent

        #######################################################
        # PRODUCT
        #######################################################

        self.product_name = product_config[
            "product_name"
        ]

        #######################################################
        # DATA FILE
        #######################################################

        configured_data_file = Path(
            product_config["data_file"]
        )

        if configured_data_file.is_absolute():

            self.data_file = (
                configured_data_file
            )

        else:

            self.data_file = (
                self.project_root
                / configured_data_file
            )

        #######################################################
        # MODEL FOLDER
        #######################################################

        configured_model_folder = Path(
            product_config["model_folder"]
        )

        if configured_model_folder.is_absolute():

            self.model_folder = (
                configured_model_folder
            )

        else:

            self.model_folder = (
                self.project_root
                / configured_model_folder
            )

        #######################################################
        # INPUT FEATURES
        #######################################################

        self.input_features = (
            product_config[
                "input_features"
            ]
        )

        #######################################################
        # QUALITY TARGETS
        #######################################################

        self.targets = (
            product_config[
                "targets"
            ]
        )

        #######################################################
        # DISPLAY NAMES
        #######################################################

        self.display_names = (
            product_config.get(
                "display_names",
                {}
            )
        )

        #######################################################
        # QUALITY RULES
        #######################################################

        self.quality_rules = (
            product_config.get(
                "quality_rules",
                {}
            )
        )

        #######################################################
        # MINIMUM ACCEPTABLE ATTRIBUTES
        #######################################################

        self.minimum_acceptable_attributes = (
            product_config.get(
                "minimum_acceptable_attributes",
                1
            )
        )

        #######################################################
        # SHELF LIFE CONFIGURATION
        #######################################################

        shelf_config = (
            product_config.get(
                "shelf_life",
                {}
            )
        )

        self.ambient_reference_limit = (
            shelf_config.get(
                "ambient_reference_limit",
                8.0
            )
        )

        self.refrigerated_reference_limit = (
            shelf_config.get(
                "refrigerated_reference_limit",
                22.0
            )
        )

        self.temperature_q10 = (
            shelf_config.get(
                "temperature_q10",
                2.0
            )
        )

        #######################################################
        # MODEL CONTAINERS
        #######################################################

        self.models = {}

        self.scalers = {}

        #######################################################
        # EXPERIMENTAL DATA
        #######################################################

        self.experimental_data = None

        self.experimental_domain = {}

        #######################################################
        # INITIALIZATION
        #######################################################

        self.load_experimental_data()

        self.load_models()

        self.load_scalers()

        self.validate_system()


    ############################################################
    # LOAD EXPERIMENTAL DATA
    ############################################################

    def load_experimental_data(self):

        if not self.data_file.exists():

            raise FileNotFoundError(
                "Experimental data file not found:\n"
                f"{self.data_file}\n\n"
                "Please check the project data folder."
            )

        try:

            df = pd.read_excel(
                self.data_file
            )

        except Exception as e:

            raise RuntimeError(
                "Could not read experimental data file.\n"
                f"File: {self.data_file}\n"
                f"Error: {e}"
            )

        #######################################################
        # Remove unnamed columns
        #######################################################

        df = df.loc[
            :,
            ~df.columns.astype(str)
            .str.contains(
                "^Unnamed",
                regex=True
            )
        ]

        #######################################################
        # Required columns
        #######################################################

        required_columns = (
            self.input_features
            +
            self.targets
        )

        missing = [

            column
            for column in required_columns

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                "Missing required columns "
                "in experimental data:\n"
                + str(missing)
            )

        #######################################################
        # Numeric conversion
        #######################################################

        for column in required_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        #######################################################
        # Keep valid input rows
        #######################################################

        df = df.dropna(
            subset=self.input_features
        )

        if df.empty:

            raise ValueError(
                "Experimental dataset contains "
                "no valid input rows."
            )

        #######################################################
        # Sort
        #######################################################

        df = df.sort_values(
            by=self.input_features
        ).reset_index(
            drop=True
        )

        self.experimental_data = df

        #######################################################
        # Experimental domain
        #######################################################

        for feature in self.input_features:

            values = (
                df[feature]
                .dropna()
                .unique()
                .tolist()
            )

            if not values:

                raise RuntimeError(
                    f"No experimental values found "
                    f"for {feature}."
                )

            self.experimental_domain[
                feature
            ] = sorted(values)


    ############################################################
    # FIND MODEL FILE
    ############################################################

    def find_model_file(
        self,
        target
    ):

        candidates = [

            f"{target}_model.pkl",

            f"{target}_best_model.pkl",

            f"best_model_{target}.pkl",

            f"model_{target}.pkl",

            f"{target}.pkl"

        ]

        #######################################################
        # Direct search
        #######################################################

        for filename in candidates:

            path = (
                self.model_folder
                / filename
            )

            if path.exists():

                return path


        #######################################################
        # Recursive search
        #######################################################

        target_lower = target.lower()

        if self.model_folder.exists():

            for path in self.model_folder.rglob(
                "*.pkl"
            ):

                filename = (
                    path.name.lower()
                )

                if target_lower in filename:

                    return path

        return None


    ############################################################
    # FIND SCALER FILE
    ############################################################

    def find_scaler_file(
        self,
        target
    ):

        candidates = [

            f"{target}_scaler.pkl",

            f"scaler_{target}.pkl",

            f"{target}_standard_scaler.pkl"

        ]

        #######################################################
        # Direct search
        #######################################################

        for filename in candidates:

            path = (
                self.model_folder
                / filename
            )

            if path.exists():

                return path


        #######################################################
        # Recursive search
        #######################################################

        target_lower = target.lower()

        if self.model_folder.exists():

            for path in self.model_folder.rglob(
                "*.pkl"
            ):

                filename = (
                    path.name.lower()
                )

                if (
                    "scaler" in filename
                    and target_lower in filename
                ):

                    return path

        return None


    ############################################################
    # LOAD AI MODELS
    ############################################################

    def load_models(self):

        if not self.model_folder.exists():

            raise FileNotFoundError(
                "Model folder not found:\n"
                f"{self.model_folder}"
            )

        failed = []

        for target in self.targets:

            model_file = (
                self.find_model_file(
                    target
                )
            )

            if model_file is None:

                failed.append(target)

                continue

            try:

                self.models[target] = (
                    joblib.load(
                        model_file
                    )
                )

            except Exception as e:

                print(
                    f"Could not load model "
                    f"{target}: {e}"
                )

                failed.append(target)

        #######################################################
        # At least one model required
        #######################################################

        if len(self.models) == 0:

            raise RuntimeError(
                "No AI models could be loaded.\n"
                f"Model folder: {self.model_folder}"
            )

        #######################################################
        # Informational message
        #######################################################

        if failed:

            print(
                "WARNING: The following models "
                "could not be loaded:"
            )

            for target in failed:

                print(
                    f"  - {target}"
                )


    ############################################################
    # LOAD SCALERS
    ############################################################

    def load_scalers(self):

        for target in self.models:

            scaler_file = (
                self.find_scaler_file(
                    target
                )
            )

            if scaler_file is None:

                continue

            try:

                self.scalers[target] = (
                    joblib.load(
                        scaler_file
                    )
                )

            except Exception as e:

                print(
                    f"WARNING: Could not load scaler "
                    f"for {target}: {e}"
                )


    ############################################################
    # VALIDATE SYSTEM
    ############################################################

    def validate_system(self):

        if self.experimental_data is None:

            raise RuntimeError(
                "Experimental data are not loaded."
            )

        if len(self.models) == 0:

            raise RuntimeError(
                "No trained AI models are available."
            )

        for feature in self.input_features:

            if feature not in self.experimental_domain:

                raise RuntimeError(
                    f"No experimental domain "
                    f"for {feature}."
                )


    ############################################################
    # PREPARE INPUT
    ############################################################

    def prepare_input(
        self,
        inputs
    ):

        missing = [

            feature

            for feature in self.input_features

            if feature not in inputs

        ]

        if missing:

            raise ValueError(
                "Missing input variables:\n"
                + str(missing)
            )

        return pd.DataFrame(
            [
                [
                    float(inputs[feature])
                    for feature in self.input_features
                ]
            ],
            columns=self.input_features
        )


    ############################################################
    # INPUT VALIDATION
    ############################################################

    def validate_inputs(
        self,
        inputs
    ):

        warnings_list = []

        for feature in self.input_features:

            value = float(
                inputs[feature]
            )

            domain = (
                self.experimental_domain[
                    feature
                ]
            )

            minimum = min(domain)

            maximum = max(domain)

            if value < minimum:

                warnings_list.append(
                    f"{feature}={value:g} "
                    f"is below experimental "
                    f"minimum ({minimum:g})."
                )

            elif value > maximum:

                warnings_list.append(
                    f"{feature}={value:g} "
                    f"is above experimental "
                    f"maximum ({maximum:g})."
                )


        #######################################################
        # Exact experimental combination
        #######################################################

        mask = np.ones(
            len(self.experimental_data),
            dtype=bool
        )

        for feature in self.input_features:

            mask &= np.isclose(

                self.experimental_data[
                    feature
                ].to_numpy(
                    dtype=float
                ),

                float(
                    inputs[feature]
                )

            )

        if not mask.any():

            warnings_list.append(
                "Input combination does not exactly "
                "exist in the experimental dataset. "
                "The AI model is being used for prediction."
            )

        return warnings_list


    ############################################################
    # AI PREDICTION
    ############################################################

    def predict_attribute(
        self,
        target,
        sample
    ):

        if target not in self.models:

            raise RuntimeError(
                f"No model available for {target}"
            )

        model = self.models[target]

        #######################################################
        # Apply scaler if available
        #######################################################

        if target in self.scalers:

            X = self.scalers[
                target
            ].transform(
                sample
            )

        else:

            X = sample

        #######################################################
        # Prediction
        #######################################################

        prediction = model.predict(
            X
        )

        return float(
            np.asarray(
                prediction
            ).ravel()[0]
        )


    ############################################################
    # STAGE 1 - AI PREDICTION
    ############################################################

    def stage_1_ai_prediction(
        self,
        inputs
    ):

        sample = self.prepare_input(
            inputs
        )

        predictions = {}

        warnings_list = []

        for target in self.models:

            try:

                predictions[target] = (
                    self.predict_attribute(
                        target,
                        sample
                    )
                )

            except Exception as e:

                warnings_list.append(
                    f"{target}: {str(e)}"
                )

        return (
            predictions,
            warnings_list
        )


    ############################################################
    # QUALITY RULE EVALUATION
    ############################################################

    def evaluate_attribute(
        self,
        target,
        value
    ):

        rule = (
            self.quality_rules.get(
                target
            )
        )

        if rule is None:

            return True

        rule_type = rule.get(
            "type"
        )

        if rule_type == "max":

            return (
                value <= rule["max"]
            )

        if rule_type == "min":

            return (
                value >= rule["min"]
            )

        if rule_type == "range":

            return (

                rule["min"]
                <= value
                <= rule["max"]

            )

        if rule_type == "reference":

            reference = (
                rule["reference"]
            )

            tolerance = (
                rule["tolerance"]
            )

            return (

                reference - tolerance
                <= value
                <=
                reference + tolerance

            )

        return True


    ############################################################
    # EVALUATE QUALITY
    ############################################################

    def evaluate_quality(
        self,
        predictions
    ):

        evaluation = {}

        for target, value in predictions.items():

            evaluation[target] = (
                self.evaluate_attribute(
                    target,
                    value
                )
            )

        return evaluation


    ############################################################
    # QUALITY SCORE
    ############################################################

    def calculate_quality_score(
        self,
        evaluation
    ):

        if len(evaluation) == 0:

            return (
                "UNKNOWN",
                0.0
            )

        acceptable = sum(
            bool(v)
            for v in evaluation.values()
        )

        total = len(evaluation)

        score = (
            acceptable
            /
            total
            *
            100.0
        )

        #######################################################
        # Status
        #######################################################

        if acceptable >= max(
            self.minimum_acceptable_attributes + 2,
            1
        ):

            status = "EXCELLENT"

        elif acceptable >= (
            self.minimum_acceptable_attributes
        ):

            status = "ACCEPTABLE"

        elif acceptable >= max(
            self.minimum_acceptable_attributes - 2,
            1
        ):

            status = "WARNING"

        else:

            status = "REJECT"

        return (
            status,
            score
        )


    ############################################################
    # EXPERIMENTAL SHELF LIFE
    ############################################################

    def calculate_experimental_shelf_life(
        self,
        inputs
    ):

        time_feature = (
            self.config.get(
                "time_feature"
            )
        )

        if time_feature is None:

            return None

        treatment_features = [

            feature

            for feature in self.input_features

            if feature != time_feature

        ]

        working_data = (
            self.experimental_data.copy()
        )

        #######################################################
        # Find nearest treatment conditions
        #######################################################

        for feature in treatment_features:

            value = float(
                inputs[feature]
            )

            available = (
                self.experimental_domain[
                    feature
                ]
            )

            nearest = min(

                available,

                key=lambda x:
                abs(x - value)

            )

            working_data = (
                working_data[
                    np.isclose(
                        working_data[
                            feature
                        ],
                        nearest
                    )
                ]
            )

        if working_data.empty:

            return None

        #######################################################
        # Evaluate observed times
        #######################################################

        valid_times = []

        for time_value in sorted(

            working_data[
                time_feature
            ].unique()

        ):

            day_data = (
                working_data[
                    np.isclose(
                        working_data[
                            time_feature
                        ],
                        time_value
                    )
                ]
            )

            observed = {}

            for target in self.targets:

                if target not in day_data.columns:

                    continue

                values = (
                    day_data[target]
                    .dropna()
                )

                if len(values) == 0:

                    continue

                observed[target] = (
                    float(
                        values.mean()
                    )
                )

            if not observed:

                continue

            evaluation = (
                self.evaluate_quality(
                    observed
                )
            )

            acceptable = sum(
                bool(v)
                for v in evaluation.values()
            )

            if acceptable >= (
                self.minimum_acceptable_attributes
            ):

                valid_times.append(
                    float(time_value)
                )

        if not valid_times:

            return None

        return max(
            valid_times
        )


    ############################################################
    # STAGE 2 - EXPERIMENTAL ANCHOR
    ############################################################

    def stage_2_experimental_anchor(
        self,
        inputs
    ):

        return (
            self.calculate_experimental_shelf_life(
                inputs
            )
        )


    ############################################################
    # STAGE 3 - PHYSICAL / TEMPERATURE CONSTRAINT
    ############################################################

    def stage_3_physical_constraint(
        self,
        inputs,
        anchor
    ):

        temperature_feature = (
            self.config.get(
                "temperature_feature"
            )
        )

        if temperature_feature is None:

            return anchor

        temperature = float(
            inputs[
                temperature_feature
            ]
        )

        #######################################################
        # No experimental anchor
        #######################################################

        if anchor is None:

            if temperature <= 4:

                anchor = (
                    self.refrigerated_reference_limit
                )

            else:

                anchor = (
                    self.ambient_reference_limit
                )

        #######################################################
        # Reference temperature domain
        #######################################################

        if temperature <= 25:

            if temperature <= 4:

                return min(
                    anchor,
                    self.refrigerated_reference_limit
                )

            return min(
                anchor,
                self.ambient_reference_limit
            )

        #######################################################
        # Q10 correction above 25°C
        #######################################################

        factor = (

            self.temperature_q10
            **
            (
                (temperature - 25.0)
                / 10.0
            )

        )

        adjusted = (
            anchor / factor
        )

        adjusted = min(
            adjusted,
            self.ambient_reference_limit
        )

        return max(
            1.0,
            float(adjusted)
        )


    ############################################################
    # SHELF LIFE
    ############################################################

    def estimate_shelf_life(
        self,
        inputs
    ):

        anchor = (
            self.stage_2_experimental_anchor(
                inputs
            )
        )

        constrained = (
            self.stage_3_physical_constraint(
                inputs,
                anchor
            )
        )

        time_feature = (
            self.config.get(
                "time_feature"
            )
        )

        current_time = None

        if time_feature is not None:

            current_time = float(
                inputs[
                    time_feature
                ]
            )

        if (
            constrained is not None
            and current_time is not None
        ):

            remaining = max(
                0.0,
                constrained - current_time
            )

        else:

            remaining = constrained

        return {

            "experimental_anchor":
                anchor,

            "temperature_constrained":
                constrained,

            "current_storage_time":
                current_time,

            "remaining":
                remaining

        }


    ############################################################
    # RISK
    ############################################################

    def assess_risk(
        self,
        status,
        domain_warnings
    ):

        if status == "REJECT":

            return "HIGH"

        if status == "WARNING":

            return "MEDIUM"

        if len(domain_warnings) > 0:

            return "MEDIUM"

        return "LOW"


    ############################################################
    # RECOMMENDATION
    ############################################################

    def generate_recommendation(
        self,
        status,
        risk
    ):

        if status == "EXCELLENT":

            return (
                "Excellent quality. "
                "Continue storage under controlled conditions."
            )

        if status == "ACCEPTABLE":

            return (
                "Quality is acceptable. "
                "The product is suitable for market "
                "with routine monitoring."
            )

        if status == "WARNING":

            return (
                "Quality deterioration detected. "
                "Sell soon or improve storage conditions."
            )

        return (
            "Quality is below the acceptable level. "
            "The product is not recommended for market."
        )


    ############################################################
    # MAIN PREDICTION API
    ############################################################

    def predict(
        self,
        inputs
    ):

        #######################################################
        # Validate
        #######################################################

        validation_warnings = (
            self.validate_inputs(
                inputs
            )
        )

        #######################################################
        # Stage 1
        #######################################################

        predictions, prediction_warnings = (
            self.stage_1_ai_prediction(
                inputs
            )
        )

        if not predictions:

            raise RuntimeError(
                "No quality attribute could be predicted."
            )

        #######################################################
        # Quality
        #######################################################

        evaluation = (
            self.evaluate_quality(
                predictions
            )
        )

        status, score = (
            self.calculate_quality_score(
                evaluation
            )
        )

        #######################################################
        # Shelf life
        #######################################################

        shelf_life = (
            self.estimate_shelf_life(
                inputs
            )
        )

        #######################################################
        # Current time constraint
        #######################################################

        time_feature = (
            self.config.get(
                "time_feature"
            )
        )

        current_time = None

        if time_feature is not None:

            current_time = float(
                inputs[
                    time_feature
                ]
            )

        constrained = (
            shelf_life[
                "temperature_constrained"
            ]
        )

        if (
            constrained is not None
            and current_time is not None
        ):

            if current_time > constrained:

                status = "REJECT"

                score = min(
                    score,
                    49.9
                )

            elif current_time >= (
                constrained - 1.0
            ):

                if status in [
                    "EXCELLENT",
                    "ACCEPTABLE"
                ]:

                    status = "WARNING"

        #######################################################
        # Risk
        #######################################################

        risk = (
            self.assess_risk(
                status,
                validation_warnings
            )
        )

        #######################################################
        # Recommendation
        #######################################################

        recommendation = (
            self.generate_recommendation(
                status,
                risk
            )
        )

        #######################################################
        # Final result
        #######################################################

        return {

            "product":
                self.product_name,

            "inputs":
                dict(inputs),

            "predictions":
                predictions,

            "quality_evaluation":
                evaluation,

            "quality_score":
                round(
                    score,
                    2
                ),

            "status":
                status,

            "risk":
                risk,

            "shelf_life":
                shelf_life,

            "warnings":
                validation_warnings,

            "prediction_warnings":
                prediction_warnings,

            "recommendation":
                recommendation

        }


###############################################################
# END
###############################################################
