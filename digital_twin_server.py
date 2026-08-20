###############################################################
# FOOD DIGITAL TWIN API SERVER
# CLOUD / GITHUB / RENDER VERSION
#
# Version: 2.0
#
# Architecture:
#
# User / Mobile App
#        ↓
#      HTTPS
#        ↓
#     Render
#        ↓
#    FastAPI API
#        ↓
# DigitalTwinEngine
#        ↓
# AI Models + Experimental Data
#
# IMPORTANT:
# No local Windows paths are used.
# All files are loaded relative to this project.
###############################################################

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from digital_twin_engine import DigitalTwinEngine


###############################################################
# PROJECT DIRECTORIES
###############################################################

# Directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Models directory
MODELS_DIR = BASE_DIR / "models"


###############################################################
# FILE PATHS
###############################################################

LOQUAT_DATA_FILE = (
    DATA_DIR / "loquat_data.xlsx"
)

###############################################################
# LOQUAT CONFIGURATION
###############################################################

LOQUAT_CONFIG = {

    "product_name":
        "Loquat",

    ###########################################################
    # CLOUD-RELATIVE DATA PATH
    ###########################################################

    "data_file":
        str(LOQUAT_DATA_FILE),

    ###########################################################
    # CLOUD-RELATIVE MODEL PATH
    ###########################################################

    "model_folder":
        str(MODELS_DIR),

    ###########################################################
    # INPUT VARIABLES
    ###########################################################

    "input_features": [

        "ColdPlasmaTime",
        "StorageTemperature",
        "Days"

    ],

    ###########################################################
    # QUALITY TARGETS
    ###########################################################

    "targets": [

        "Weight",
        "pH",
        "Brix",
        "TA",
        "Index",
        "Phenol",
        "Flavnoid",
        "Hardness",
        "Resilience"

    ],

    ###########################################################
    # DISPLAY NAMES
    ###########################################################

    "display_names": {

        "Weight":
            "Weight Loss (%)",

        "pH":
            "pH",

        "Brix":
            "Brix",

        "TA":
            "Titratable Acidity",

        "Index":
            "Ripening Index",

        "Phenol":
            "Phenolic Content",

        "Flavnoid":
            "Flavonoid Content",

        "Hardness":
            "Hardness",

        "Resilience":
            "Resilience"

    },

    ###########################################################
    # QUALITY RULES
    ###########################################################

    "quality_rules": {

        "Weight": {

            "type":
                "max",

            "max":
                30.0

        },

        "pH": {

            "type":
                "range",

            "min":
                3.15,

            "max":
                3.85

        },

        "Brix": {

            "type":
                "range",

            "min":
                10.0,

            "max":
                12.0

        },

        "TA": {

            "type":
                "range",

            "min":
                0.25,

            "max":
                0.55

        },

        "Index": {

            "type":
                "range",

            "min":
                14.0,

            "max":
                34.0

        },

        "Phenol": {

            "type":
                "range",

            "min":
                0.55,

            "max":
                0.85

        },

        "Flavnoid": {

            "type":
                "range",

            "min":
                0.30,

            "max":
                0.50

        },

        "Hardness": {

            "type":
                "range",

            "min":
                90.0,

            "max":
                330.0

        },

        "Resilience": {

            "type":
                "range",

            "min":
                0.13,

            "max":
                0.37

        }

    },

    ###########################################################
    # QUALITY DECISION
    ###########################################################

    "minimum_acceptable_attributes":
        6,

    ###########################################################
    # TIME FEATURE
    ###########################################################

    "time_feature":
        "Days",

    ###########################################################
    # TEMPERATURE FEATURE
    ###########################################################

    "temperature_feature":
        "StorageTemperature",

    ###########################################################
    # SHELF LIFE
    ###########################################################

    "shelf_life": {

        "ambient_reference_limit":
            8.0,

        "refrigerated_reference_limit":
            22.0,

        "temperature_q10":
            2.0

    }

}


###############################################################
# FASTAPI APPLICATION
###############################################################

api = FastAPI(

    title=
        "Food Digital Twin API",

    description=(
        "Cloud-based AI Digital Twin "
        "for food quality and shelf-life prediction."
    ),

    version=
        "2.0.0"

)


###############################################################
# GLOBAL ENGINE VARIABLES
###############################################################

twin = None

ENGINE_READY = False

ENGINE_ERROR = None


###############################################################
# LOAD DIGITAL TWIN ENGINE
###############################################################

def initialize_engine():

    global twin
    global ENGINE_READY
    global ENGINE_ERROR

    try:

        print("=" * 70)

        print(
            "FOOD DIGITAL TWIN - ENGINE INITIALIZATION"
        )

        print("=" * 70)

        print(
            f"BASE_DIR   : {BASE_DIR}"
        )

        print(
            f"DATA_DIR   : {DATA_DIR}"
        )

        print(
            f"MODELS_DIR : {MODELS_DIR}"
        )

        print(
            f"DATA FILE  : {LOQUAT_DATA_FILE}"
        )

        #######################################################
        # Check data file
        #######################################################

        if not LOQUAT_DATA_FILE.exists():

            raise FileNotFoundError(

                "Experimental data file not found:\n"
                f"{LOQUAT_DATA_FILE}\n\n"
                "Expected location inside GitHub repository:\n"
                "data/loquat_data.xlsx"

            )

        #######################################################
        # Check model directory
        #######################################################

        if not MODELS_DIR.exists():

            raise FileNotFoundError(

                "Models directory not found:\n"
                f"{MODELS_DIR}\n\n"
                "Expected location inside GitHub repository:\n"
                "models/"

            )

        #######################################################
        # Initialize engine
        #######################################################

        twin = DigitalTwinEngine(
            LOQUAT_CONFIG
        )

        ENGINE_READY = True

        ENGINE_ERROR = None

        print(
            "Digital Twin Engine: READY"
        )

        print("=" * 70)

    except Exception as e:

        twin = None

        ENGINE_READY = False

        ENGINE_ERROR = (
            f"{type(e).__name__}: {str(e)}"
        )

        print("=" * 70)

        print(
            "Digital Twin Engine: ERROR"
        )

        print(
            ENGINE_ERROR
        )

        print("=" * 70)


###############################################################
# INITIALIZE ENGINE
###############################################################

initialize_engine()


###############################################################
# REQUEST MODEL
###############################################################

class PredictionRequest(BaseModel):

    product: str = Field(

        default="Loquat",

        description=
            "Product name"

    )

    ColdPlasmaTime: float = Field(

        ...,

        ge=0,

        description=
            "Cold plasma treatment time in minutes"

    )

    StorageTemperature: float = Field(

        ...,

        description=
            "Storage temperature in Celsius"

    )

    Days: float = Field(

        ...,

        ge=0,

        description=
            "Storage time in days"

    )


###############################################################
# ROOT ENDPOINT
###############################################################

@api.get("/")
def home():

    return {

        "application":
            "Food Digital Twin",

        "version":
            "2.0.0",

        "status":
            "online",

        "engine_ready":
            ENGINE_READY,

        "product":
            "Loquat"

    }


###############################################################
# HEALTH CHECK
###############################################################

@api.get("/health")
def health():

    if not ENGINE_READY:

        return {

            "status":
                "error",

            "engine_ready":
                False,

            "error":
                ENGINE_ERROR

        }

    return {

        "status":
            "healthy",

        "engine_ready":
            True,

        "product":
            "Loquat"

    }


###############################################################
# PRODUCT ENDPOINT
###############################################################

@api.get("/products")
def products():

    return {

        "products": [

            {

                "id":
                    "loquat",

                "name":
                    "Loquat",

                "status":
                    "available"

            }

        ]

    }


###############################################################
# MODEL INFORMATION
###############################################################

@api.get("/model-info")
def model_info():

    if not ENGINE_READY:

        return {

            "engine_ready":
                False,

            "error":
                ENGINE_ERROR

        }

    return {

        "engine_ready":
            True,

        "product":
            twin.product_name,

        "input_features":
            twin.input_features,

        "targets":
            list(twin.models.keys()),

        "experimental_data_available":
            twin.experimental_data is not None,

        "model_count":
            len(twin.models)

    }


###############################################################
# PREDICTION ENDPOINT
###############################################################

@api.post("/predict")
def predict(
    request: PredictionRequest
):

    ###########################################################
    # ENGINE CHECK
    ###########################################################

    if not ENGINE_READY:

        raise HTTPException(

            status_code=503,

            detail={

                "message":
                    "Digital Twin Engine is not ready.",

                "error":
                    ENGINE_ERROR

            }

        )

    ###########################################################
    # PRODUCT CHECK
    ###########################################################

    if request.product.lower() != "loquat":

        raise HTTPException(

            status_code=400,

            detail=(
                "Currently only Loquat "
                "is supported."
            )

        )

    ###########################################################
    # CREATE INPUT
    ###########################################################

    inputs = {

        "ColdPlasmaTime":
            request.ColdPlasmaTime,

        "StorageTemperature":
            request.StorageTemperature,

        "Days":
            request.Days

    }

    ###########################################################
    # RUN DIGITAL TWIN
    ###########################################################

    try:

        result = twin.predict(
            inputs
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail={

                "message":
                    "Digital Twin prediction failed.",

                "error":
                    str(e)

            }

        )

    ###########################################################
    # RETURN RESULT
    ###########################################################

    return {

        "success":
            True,

        "product":
            result["product"],

        "inputs":
            result["inputs"],

        "predictions":
            result["predictions"],

        "quality_evaluation":
            result["quality_evaluation"],

        "quality_score":
            result["quality_score"],

        "status":
            result["status"],

        "risk":
            result["risk"],

        "shelf_life":
            result["shelf_life"],

        "warnings":
            result["warnings"],

        "prediction_warnings":
            result["prediction_warnings"],

        "recommendation":
            result["recommendation"]

    }


###############################################################
# RELOAD ENGINE
#
# Useful for administration / testing.
###############################################################

@api.post("/reload")
def reload_engine():

    initialize_engine()

    return {

        "engine_ready":
            ENGINE_READY,

        "error":
            ENGINE_ERROR

    }


###############################################################
# SERVER START
###############################################################

if __name__ == "__main__":

    ###########################################################
    # Render provides PORT as environment variable.
    #
    # Local default = 8000
    ###########################################################

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    ###########################################################
    # IMPORTANT:
    #
    # 0.0.0.0 allows Render / cloud traffic
    # to reach the application.
    ###########################################################

    uvicorn.run(

        "digital_twin_server:api",

        host=
            "0.0.0.0",

        port=
            port,

        reload=
            False

    )
