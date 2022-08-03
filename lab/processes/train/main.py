"""
This file contains the code for launch the train step
"""
import logging

import pandas as pd

from processes.config import ConfigPreprocess
from processes.train.train import train

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Load dataset
    logger.info("Training with %s", ConfigPreprocess.PREPROCESS_FILE)
    df_preprocess = pd.read_csv(ConfigPreprocess.PREPROCESS_FILE)

    # Train model
    train(df=df_preprocess)
