###############################################################
# FOOD DIGITAL TWIN API SERVER
# VERSION 2.0
#
# Portable / Local Server Version
#
# The server automatically searches for:
#   1. Experimental Excel data
#   2. AI model files
#
# No fixed F:\... path is required.
###############################################################

import os
import glob
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from digital_twin_engine import DigitalTwinEngine


###############################################################
# PROJECT PATH
###############################################################

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

print("\n" + "=" * 70)
print("FOOD DIGITAL TWIN SERVER")
print("=" * 70)

print("\nAPP DIRECTORY:")
print(BASE_DIR)


###############################################################
# FIND PROJECT ROOT
###############################################################

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

print("\nPROJECT ROOT:")
print(PROJECT_ROOT)


###############################################################
# SEARCH FUNCTION
###############################################################

def find_file_recursive(
    root,
    extensions=None,
    keywords=None
):

    if extensions is None:
        extensions = []

    if keywords is None:
        keywords = []

    matches = []

    for current_root, dirs, files in os.walk(root):

        for filename in files:

            lower_name = filename.lower()

            ###################################################
            # Extension filter
            ###################################################

            if extensions:

                if not any(
                    lower_name.endswith(ext.lower())
                    for ext in extensions
                ):

                    continue

            ###################################################
            # Keyword filter
            ###################################################

            if keywords:

                if not all(
                    keyword.lower() in lower_name
                    for keyword in keywords
                ):

                    continue

            matches.append(
                os.path.join(
                    current_root,
                    filename
                )
            )

    return matches


###############################################################
# FIND EXPERIMENTAL DATA
###############################################################

def find_experimental_data():

    ###########################################################
    # First priority:
    # common Excel names
    ###########################################################

    preferred_names = [

        "loquat_data.xlsx",
        "loquat_data.xls",
        "Loquat.xlsx",
        "Loquat.xls",
        "Loquat(5).xlsx",
        "Loquat(5).xls"

    ]

    search_roots = [

        BASE_DIR,
        PROJECT_ROOT

    ]

    ###########################################################
    # Search preferred names
    ###########################################################

    for root in search_roots:

        for preferred in preferred_names:

            matches = glob.glob(
                os.path.join(
                    root,
                    "**",
                    preferred
                ),
                recursive=True
            )

            if matches:

                return os.path.abspath(
                    matches[0]
                )

    ###########################################################
    # Search Excel files containing loquat
    ###########################################################

    for root in search_roots:

        matches = find_file_recursive(

            root,

            extensions=[
                ".xlsx",
                ".xls"
            ],

            keywords=[
                "loquat"
            ]

        )

        if matches:

            return os.path.abspath(
                matches[0]
            )

    ###########################################################
    # Search any Excel file as last resort
    ###########################################################

    for root in search_roots:

        matches = find_file_recursive(

            root,

            extensions=[
                ".xlsx",
                ".xls"
            ]

        )

        if matches:

            return os.path.abspath(
                matches[0]
            )

    return None


###############################################################
# FIND MODEL DIRECTORY
###############################################################

def find_model_directory():

    ###########################################################
    # Preferred model folders
    ###########################################################

    preferred = [

        os.path.join(
            BASE_DIR,
            "MODELS"
        ),

        os.path.join(
            BASE_DIR,
            "models"
        ),

        os.path.join(
            PROJECT_ROOT,
            "MODELS"
        ),

        os.path.join(
            PROJECT_ROOT,
            "models"
        )

    ]

    for folder in preferred:

        if os.path.isdir(folder):

            pkl_files = glob.glob(
                os.path.join(
                    folder,
                    "**",
                    "*.pkl"
                ),
                recursive=True
            )

            if pkl_files:

                return os.path.abspath(
                    folder
                )

    ###########################################################
    # Search recursively for folders containing pkl
    ###########################################################

    for root, dirs, files in os.walk(
        PROJECT_ROOT
    ):

        pkl_files = [

            f for f in files

            if f.lower().endswith(
                ".pkl"
            )

        ]

        if pkl_files:

            return os.path.abspath(
                root
            )

    return None


###############################################################
# LOCATE DATA
###############################################################

DATA_FILE = find_experimental_data()

MODEL_FOLDER = find_model_directory()


print("\n" + "-" * 70)

print("EXPERIMENTAL DATA:")

if DATA_FILE:

    print(DATA_FILE)

else:

    print("NOT FOUND")


print("\nMODEL FOLDER:")

if MODEL_FOLDER:

    print(MODEL_FOLDER)

else:

    print("NOT FOUND")

print("-" * 70)


###############################################################
# LOQUAT CONFIGURATION
###############################################################

LOQUAT_CONFIG = {

    "product_name":
        "Loquat",

    "data_file":
        DATA_FILE,

    "model_folder":
        MODEL_FOLDER,

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
    # TIME / TEMPERATURE
    ###########################################################

    "time_feature":
        "Days",

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

    description=
        "AI-based Quality and Shelf-Life Prediction Engine",

    version=
        "2.0.0"

)


###############################################################
# ENGINE INITIALIZATION
###############################################################

twin = None

ENGINE_READY = False

ENGINE_ERROR = None


def initialize_engine():

    global twin

    global ENGINE_READY

    global ENGINE_ERROR

    ###########################################################
    # Reset
    ###########################################################

    twin = None

    ENGINE_READY = False

    ENGINE_ERROR = None

    ###########################################################
    # Check data
    ###########################################################

    if not DATA_FILE:

        ENGINE_ERROR = (

            "Experimental data file could not be found.\n\n"

            "Server directory:\n"

            + BASE_DIR

        )

        return

    ###########################################################
    # Check models
    ###########################################################

    if not MODEL_FOLDER:

        ENGINE_ERROR = (

            "AI model folder could not be found.\n\n"

            "Server directory:\n"

            + BASE_DIR

        )

        return

    ###########################################################
    # Create engine
    ###########################################################

    try:

        twin = DigitalTwinEngine(
            LOQUAT_CONFIG
        )

        ENGINE_READY = True

        ENGINE_ERROR = None

        print("\nDIGITAL TWIN ENGINE: READY")

    except Exception as e:

        ENGINE_READY = False

        ENGINE_ERROR = (

            type(e).__name__

            + ": "

            + str(e)

        )

        print("\nDIGITAL TWIN ENGINE: ERROR")

        print(ENGINE_ERROR)

        print("\nFULL ERROR:")

        traceback.print_exc()


###############################################################
# INITIALIZE
###############################################################

initialize_engine()


###############################################################
# REQUEST MODEL
###############################################################

class PredictionRequest(
    BaseModel
):

    product: str = Field(

        default="Loquat"

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
# ROOT
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

        "data_file":
            DATA_FILE,

        "model_folder":
            MODEL_FOLDER,

        "engine_error":
            ENGINE_ERROR

    }


###############################################################
# HEALTH
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
            True

    }


###############################################################
# PRODUCTS
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
# PREDICT
###############################################################

@api.post("/predict")
def predict(

    request:
        PredictionRequest

):

    ###########################################################
    # ENGINE CHECK
    ###########################################################

    if not ENGINE_READY:

        raise HTTPException(

            status_code=500,

            detail={

                "message":
                    "Digital Twin Engine is not ready.",

                "error":
                    ENGINE_ERROR,

                "data_file":
                    DATA_FILE,

                "model_folder":
                    MODEL_FOLDER

            }

        )

    ###########################################################
    # PRODUCT CHECK
    ###########################################################

    if request.product.lower() != "loquat":

        raise HTTPException(

            status_code=400,

            detail=
                "Currently only Loquat is supported."

        )

    ###########################################################
    # INPUT
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
    # RUN ENGINE
    ###########################################################

    try:

        result = twin.predict(
            inputs
        )

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    ###########################################################
    # RESPONSE
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
###############################################################

@api.post("/reload")
def reload_engine():

    initialize_engine()

    return {

        "engine_ready":
            ENGINE_READY,

        "data_file":
            DATA_FILE,

        "model_folder":
            MODEL_FOLDER,

        "error":
            ENGINE_ERROR

    }


###############################################################
# SERVER
###############################################################

if __name__ == "__main__":

    uvicorn.run(

        "digital_twin_server:api",

        host=
            "127.0.0.1",

        port=
            8000,

        reload=
            False

    )
