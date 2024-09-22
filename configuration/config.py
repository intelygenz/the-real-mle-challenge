from pathlib import Path

from logger import logger

# Define directories
DIR_REPO = Path.cwd()
logger.info(f"Repository directory: {DIR_REPO}")
DIR_DATA_PROCESSED = Path(DIR_REPO) / "data" / "processed"
DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
DIR_MODELS = Path(DIR_REPO) / "models"
TESTS_DIR = Path(DIR_REPO) / "tests"

# Define file paths
FILEPATH_PROCESSED = DIR_DATA_PROCESSED / "preprocessed_listings_guillermo"
FILEPATH_MODEL = DIR_MODELS / "simple_classifier_guillermo.pkl"
FILEPATH_DATA = DIR_DATA_RAW / "listings.csv"
FILEPATH_TEST_INPUT_DATA = TESTS_DIR / "listings.csv"
FILEPATH_TEST_EXPECTED_DATA = TESTS_DIR / "expected_preprocessed_listings.csv"