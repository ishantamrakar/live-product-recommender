"""
Shared setup for all experiments — data loading and vectorisation.
Centralising this ensures every experiment uses identical splits and features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RESULTS_DIR   = Path(__file__).parent.parent.parent / "results"


def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, val, test


def build_vectoriser(train_text, max_features=20_000):
    vec = CountVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train = vec.fit_transform(train_text)
    vocab   = np.array(vec.get_feature_names_out())
    return vec, X_train, vocab
