"""
This file contains the code for launch the preprocess
"""

import pandas as pd
from config import ConfigPreprocess
from preprocess import preprocess

if __name__ == "__main__":

    # Load dataset
    df_raw = pd.read_csv(ConfigPreprocess.RAW_FILE)

    # Preprocess dataset
    preprocess(df=df_raw, preprocess_path=ConfigPreprocess.PREPROCESS_FILE)
