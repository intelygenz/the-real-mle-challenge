"""
This file contains the code for launch the preprocess
"""
import logging

import pandas as pd

from processes.preprocess.config import ConfigPreprocess
from processes.preprocess.preprocess import preprocess

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Load dataset
    logger.info("Preprocessing %s", ConfigPreprocess.RAW_FILE)
    df_raw = pd.read_csv(ConfigPreprocess.RAW_FILE)

    # Preprocess dataset
    preprocess(df=df_raw, preprocess_path=ConfigPreprocess.PREPROCESS_FILE)
