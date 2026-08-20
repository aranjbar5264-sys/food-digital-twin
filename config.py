import os

# ============================================================
# FOOD DIGITAL TWIN - CLOUD CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# EXPERIMENTAL DATA
# ============================================================

DATA_FILE = os.path.join(
    BASE_DIR,
    "Loquat.xlsx"
)


# ============================================================
# AI MODELS
# ============================================================

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "Models"
)


# ============================================================
# INPUT COLUMNS
# ============================================================

PLASMA_COLUMN = "ColdPlasmaTime"

TEMPERATURE_COLUMN = "StorageTemperature"

TIME_COLUMN = "Days"


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"Experimental data file not found: {DATA_FILE}"
    )


if not os.path.exists(OUTPUT_FOLDER):

    raise FileNotFoundError(
        f"Models folder not found: {OUTPUT_FOLDER}"
    )
