###############################################################
# FOOD DIGITAL TWIN ENGINE
# CLOUD / GITHUB / RENDER VERSION
#
# Version: 2.0
#
# Architecture:
#
#       Cloud API
#           ↓
#   DigitalTwinEngine
#           ↓
#     AI Models (.pkl)
#           ↓
# Experimental Data (.xlsx)
#           ↓
# Quality Prediction
#           ↓
# Shelf-Life Prediction
###############################################################

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


###############################################################
# DIGITAL TWIN ENGINE
###############################################################

class DigitalTwinEngine:

    ###########################################################
    # CONSTRUCTOR
    ###########################################################

    def __init__(self, product_config):

        self.config = product_config

        self.product_name = product_config["product_name"]

        #######################################################
        # Resolve paths
        #
        # Cloud deployment:
        # All files are located relative to the application
        # directory / GitHub repository.
        #######################################################

        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.data_file = self._resolve_path(
            product_config["data_file"]
        )

        self.model_folder = self._resolve_path(
            product_config["model_folder"]
        )

        #######################################################
        # Input variables
        #######################################################

        self.input_features = product_config[
            "input_features"
        ]

        #######################################################
        # Quality targets
        #######################################################

        self.targets = product_config[
            "targets"
        ]

        #######################################################
        # Display names
        #######################################################

        self.display_names = product_config.get(
            "display_names",
            {}
        )

        #######################################################
        # Quality rules
        #######################################################

        self.quality_rules = product_config.get(
            "quality_rules",
            {}
        )

        #######################################################
        # Minimum acceptable attributes
        #######################################################

        self.minimum_acceptable_attributes = (
            product_config.get(
                "minimum_acceptable_attributes",
                1
            )
        )

        #######################################################
        # Shelf-life configuration
        #######################################################

        shelf_config = product_config.get(
            "shelf_life",
            {}
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
        # Model containers
        #######################################################

        self.models = {}

        self.scalers = {}

        #######################################################
        # Experimental data
        #######################################################

        self.experimental_data = None

        self.experimental_domain = {}

        #######################################################
        # Initialize
        #######################################################

        self.load_experimental_data()

        self.load_models()

        self.load_scalers()

        self.validate_system()


    ###########################################################
    # PATH RESOLUTION
    ###########################################################

    def _resolve_path(self, path):

        """
        Convert relative paths to paths relative to this
        Python application.

        Examples:

            data/loquat_data.xlsx

        becomes:

            <repository>/data/loquat_data.xlsx

        Absolute paths are preserved.
        """

        if os.path.isabs(path):

            return path

        return os.path.join(
            self.base_dir,
            path
        )


    ###########################################################
    # LOAD EXPERIMENTAL DATA
    ###########################################################

    def load_experimental_data(self):

        if not os.path.exists(self.data_file):

            raise FileNotFoundError(
                "Experimental data file not found:\n"
                f"{self.data_file}\n\n"
                "Please check the GitHub repository structure."
            )

        try:

            df = pd.read_excel(
                self.data_file
            )

        except Exception as e:

            raise RuntimeError(
                "Could not read experimental data file:\n"
                f"{self.data_file}\n\n"
                f"Original error: {e}"
            )

        #######################################################
        # Remove unnamed columns
        #######################################################

        df = df.loc[
            :,
            ~df.columns.astype(str)
            .str.contains(
                "^Unnamed"
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
                "Missing required columns in experimental data:\n"
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
        # Remove invalid input rows
        #######################################################

        df = df.dropna(
            subset=self.input_features
        )

        if df.empty:

            raise ValueError(
                "Experimental data contains no valid rows."
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

            self.experimental_domain[
                feature
            ] = sorted(
                values
            )


    ###########################################################
    # FIND MODEL FILE
    ###########################################################

    def find_model_file(self, target):

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

            path = os.path.join(
                self.model_folder,
                filename
            )

            if os.path.exists(path):

                return path

        #######################################################
        # Recursive search
        #######################################################

        if os.path.exists(
            self.model_folder
        ):

            target_lower = target.lower()

            for root, dirs, files in os.walk(
                self.model_folder
            ):

                for file in files:

                    lower = file.lower()

                    if (

                        target_lower in lower

                        and

                        lower.endswith(".pkl")

                    ):

                        return os.path.join(
                            root,
                            file
                        )

        return None


    ###########################################################
    # FIND SCALER FILE
    ###########################################################

    def find_scaler_file(self, target):

        candidates = [

            f"{target}_scaler.pkl",

            f"scaler_{target}.pkl",

            f"{target}_standard_scaler.pkl"

        ]

        #######################################################
        # Direct search
        #######################################################

        for filename in candidates:

            path = os.path.join(
                self.model_folder,
                filename
            )

            if os.path.exists(path):

                return path

        #######################################################
        # Recursive search
        #######################################################

        if os.path.exists(
            self.model_folder
        ):

            target_lower = target.lower()

            for root, dirs, files in os.walk(
                self.model_folder
            ):

                for file in files:

                    lower = file.lower()

                    if (

                        "scaler" in lower

                        and

                        target_lower in lower

                        and

                        lower.endswith(".pkl")

                    ):

                        return os.path.join(
                            root,
                            file
                        )

        return None


    ###########################################################
    # LOAD AI MODELS
    ###########################################################

    def load_models(self):

        failed = []

        for target in self.targets:

            model_file = self.find_model_file(
                target
            )

            if model_file is None:

                failed.append(
                    target
                )

                continue

            try:

                self.models[target] = joblib.load(
                    model_file
                )

            except Exception as e:

                print(
                    f"Could not load model "
                    f"{target}: {e}"
                )

                failed.append(
                    target
                )

        #######################################################
        # IMPORTANT
        #
        # We allow partial model loading.
        #
        # This makes the cloud API more robust.
        #######################################################

        if len(self.models) == 0:

            raise RuntimeError(
                "No AI models could be loaded.\n"
                f"Model directory: {self.model_folder}\n"
                f"Expected targets: {self.targets}"
            )


    ###########################################################
    # LOAD SCALERS
    ###########################################################

    def load_scalers(self):

        for target in self.models:

            scaler_file = self.find_scaler_file(
                target
            )

            if scaler_file is None:

                continue

            try:

                self.scalers[target] = joblib.load(
                    scaler_file
                )

            except Exception as e:

                print(
                    f"Could not load scaler "
                    f"{target}: {e}"
                )


    ###########################################################
    # VALIDATE SYSTEM
    ###########################################################

    def validate_system(self):

        if self.experimental_data is None:

            raise RuntimeError(
                "Experimental data are not loaded."
            )

        if len(self.models) == 0:

            raise RuntimeError(
                "No trained AI models are available."
            )

        #######################################################
        # Validate experimental domain
        #######################################################

        for feature in self.input_features:

            if feature not in self.experimental_domain:

                raise RuntimeError(
                    f"No experimental domain "
                    f"available for {feature}"
                )


    ###########################################################
    # PREPARE INPUT
    ###########################################################

    def prepare_input(self, inputs):

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

        #######################################################
        # Numeric conversion
        #######################################################

        values = {}

        for feature in self.input_features:

            try:

                values[feature] = float(
                    inputs[feature]
                )

            except Exception:

                raise ValueError(
                    f"Invalid numeric value for "
                    f"{feature}: "
                    f"{inputs[feature]}"
                )

        return pd.DataFrame(
            [values],
            columns=self.input_features
        )


    ###########################################################
    # INPUT VALIDATION
    ###########################################################

    def validate_inputs(self, inputs):

        warnings_list = []

        #######################################################
        # Domain validation
        #######################################################

        for feature in self.input_features:

            value = float(
                inputs[feature]
            )

            domain = self.experimental_domain[
                feature
            ]

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
        # Check exact experimental combination
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


    ###########################################################
    # AI PREDICTION
    ###########################################################

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

            X = self.scalers[target].transform(
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


    ###########################################################
    # STAGE 1
    # AI PREDICTION
    ###########################################################

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


    ###########################################################
    # QUALITY RULE EVALUATION
    ###########################################################

    def evaluate_attribute(
        self,
        target,
        value
    ):

        rule = self.quality_rules.get(
            target
        )

        if rule is None:

            return True

        rule_type = rule.get(
            "type"
        )

        #######################################################
        # Maximum
        #######################################################

        if rule_type == "max":

            return (
                value <= rule["max"]
            )

        #######################################################
        # Minimum
        #######################################################

        if rule_type == "min":

            return (
                value >= rule["min"]
            )

        #######################################################
        # Range
        #######################################################

        if rule_type == "range":

            return (

                rule["min"]
                <=
                value
                <=
                rule["max"]

            )

        #######################################################
        # Reference ± tolerance
        #######################################################

        if rule_type == "reference":

            reference = rule["reference"]

            tolerance = rule["tolerance"]

            return (

                reference - tolerance
                <=
                value
                <=
                reference + tolerance

            )

        #######################################################
        # Unknown rule
        #######################################################

        return True


    ###########################################################
    # EVALUATE QUALITY
    ###########################################################

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


    ###########################################################
    # QUALITY SCORE
    ###########################################################

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
            evaluation.values()
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

        threshold = min(
            self.minimum_acceptable_attributes,
            total
        )

        if acceptable >= threshold:

            if score >= 85:

                status = "EXCELLENT"

            else:

                status = "ACCEPTABLE"

        elif acceptable >= max(
            threshold - 2,
            1
        ):

            status = "WARNING"

        else:

            status = "REJECT"

        return (
            status,
            score
        )


    ###########################################################
    # EXPERIMENTAL SHELF LIFE
    ###########################################################

    def calculate_experimental_shelf_life(
        self,
        inputs
    ):

        time_feature = self.config.get(
            "time_feature"
        )

        if time_feature is None:

            return None

        #######################################################
        # Treatment features
        #######################################################

        treatment_features = [

            feature

            for feature in self.input_features

            if feature != time_feature

        ]

        working_data = (
            self.experimental_data.copy()
        )

        #######################################################
        # Select nearest experimental treatment
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
                        working_data[feature],
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

        time_values = sorted(

            working_data[
                time_feature
            ].dropna().unique()

        )

        for time_value in time_values:

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

                observed[target] = float(
                    values.mean()
                )

            if not observed:

                continue

            evaluation = (
                self.evaluate_quality(
                    observed
                )
            )

            acceptable = sum(
                evaluation.values()
            )

            if acceptable >= min(

                self.minimum_acceptable_attributes,

                len(evaluation)

            ):

                valid_times.append(
                    float(time_value)
                )

        if not valid_times:

            return None

        return max(
            valid_times
        )


    ###########################################################
    # STAGE 2
    # EXPERIMENTAL ANCHOR
    ###########################################################

    def stage_2_experimental_anchor(
        self,
        inputs
    ):

        return (
            self.calculate_experimental_shelf_life(
                inputs
            )
        )


    ###########################################################
    # STAGE 3
    # PHYSICAL / TEMPERATURE CONSTRAINT
    ###########################################################

    def stage_3_physical_constraint(
        self,
        inputs,
        anchor
    ):

        temperature_feature = self.config.get(
            "temperature_feature"
        )

        if temperature_feature is None:

            return anchor

        temperature = float(
            inputs[
                temperature_feature
            ]
        )

        #######################################################
        # If experimental anchor unavailable
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
        # Refrigerated conditions
        #######################################################

        if temperature <= 4:

            return min(

                anchor,

                self.refrigerated_reference_limit

            )

        #######################################################
        # Ambient / moderate temperature
        #######################################################

        if temperature <= 25:

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
                /
                10.0
            )

        )

        adjusted = anchor / factor

        adjusted = min(

            adjusted,

            self.ambient_reference_limit

        )

        return max(

            1.0,

            float(adjusted)

        )


    ###########################################################
    # SHELF LIFE ESTIMATION
    ###########################################################

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

        time_feature = self.config.get(
            "time_feature"
        )

        current_time = None

        if time_feature is not None:

            current_time = float(
                inputs[
                    time_feature
                ]
            )

        #######################################################
        # Remaining shelf life
        #######################################################

        if (

            constrained is not None

            and

            current_time is not None

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


    ###########################################################
    # RISK ASSESSMENT
    ###########################################################

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


    ###########################################################
    # RECOMMENDATION
    ###########################################################

    def generate_recommendation(
        self,
        status,
        risk
    ):

        if status == "EXCELLENT":

            return (
                "Excellent quality. "
                "The product can remain under "
                "controlled storage conditions."
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
                "The product should be sold soon or "
                "storage conditions should be improved."
            )

        return (
            "Quality is below the acceptable level. "
            "The product is not recommended for market."
        )


    ###########################################################
    # MAIN PREDICTION API
    ###########################################################

    def predict(
        self,
        inputs
    ):

        #######################################################
        # Validate input values
        #######################################################

        validation_warnings = (
            self.validate_inputs(
                inputs
            )
        )

        #######################################################
        # Stage 1: AI prediction
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
        # Quality evaluation
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
        # Shelf-life estimation
        #######################################################

        shelf_life = (
            self.estimate_shelf_life(
                inputs
            )
        )

        #######################################################
        # Current time constraint
        #######################################################

        time_feature = self.config.get(
            "time_feature"
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

        #######################################################
        # Shelf-life status
        #######################################################

        if (

            constrained is not None

            and

            current_time is not None

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
# END OF CLOUD DIGITAL TWIN ENGINE
###############################################################
