import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_raw():
    reviews = pd.read_json(RAW_DIR / "reviews_electronics.jsonl", lines=True)
    meta    = pd.read_json(RAW_DIR / "meta_electronics.jsonl", lines=True)
    return reviews, meta


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)       # strip HTML tags
    text = re.sub(r"[^a-zA-Z\s']", " ", text)  # keep only letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_reviews(df):
    df = df.copy()

    df = df.drop(columns=["images", "asin"])
    df = df.rename(columns={"text": "review_text"})

    df["review_text"] = df["review_text"].fillna("").apply(clean_text)
    df["title"]       = df["title"].fillna("").apply(clean_text)

    df["liked"]       = (df["rating"] >= 4).astype(int)
    df["review_len"]  = df["review_text"].str.split().str.len()
    df["verified_purchase"] = df["verified_purchase"].astype(int)
    df["timestamp"]   = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def clean_metadata(df):
    df = df.copy()

    df = df.drop(columns=["bought_together", "subtitle", "author", "videos", "images"],
                 errors="ignore")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    price_median = df["price"].median()
    df["price_known"] = df["price"].notna().astype(int)
    df["price"]       = df["price"].fillna(price_median)
    df["log_price"]   = np.log1p(df["price"])

    df["top_category"] = df["categories"].apply(
        lambda c: c[1] if isinstance(c, list) and len(c) > 1 else "Unknown"
    )

    df["store"] = df["store"].fillna("Unknown")
    df["main_category"] = df["main_category"].fillna("Unknown")

    df = df.drop(columns=["categories", "description", "features", "details"],
                 errors="ignore")

    return df


def merge(reviews, meta):
    reviews = reviews.rename(columns={"title": "review_title"})
    meta    = meta.rename(columns={"title": "product_title"})
    df = reviews.merge(meta, on="parent_asin", how="left")

    # Fill metadata columns that didn't match
    df["price_known"]   = df["price_known"].fillna(0).astype(int)
    df["log_price"]     = df["log_price"].fillna(np.log1p(meta["price"].median()))
    df["average_rating"] = df["average_rating"].fillna(meta["average_rating"].median())
    df["rating_number"] = df["rating_number"].fillna(0)
    df["top_category"]  = df["top_category"].fillna("Unknown")
    df["store"]         = df["store"].fillna("Unknown")

    return df


def split(df, val_size=0.15, test_size=0.15, seed=42):
    # Chronological split — avoids data leakage from future reviews
    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)

    val_start  = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))

    train = df.iloc[:val_start]
    val   = df.iloc[val_start:test_start]
    test  = df.iloc[test_start:]

    return train, val, test


def run():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    reviews, meta = load_raw()
    print(f"  Reviews : {len(reviews):,}")
    print(f"  Metadata: {len(meta):,}")

    print("\nCleaning...")
    reviews = clean_reviews(reviews)
    meta    = clean_metadata(meta)

    print("Merging...")
    df = merge(reviews, meta)
    print(f"  Merged  : {len(df):,} rows × {df.shape[1]} cols")

    print("Splitting (chronological)...")
    train, val, test = split(df)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    print("\nSaving...")
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet",   index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print("✓ Done — saved to data/processed/")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nClass balance (train):")
    print(train["liked"].value_counts(normalize=True).to_string())


if __name__ == "__main__":
    run()
