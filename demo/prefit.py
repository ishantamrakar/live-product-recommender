"""
Run once before launching the demo.
Fits the HybridNB model on training data and saves all artifacts
the Streamlit app needs at startup.

    python demo/prefit.py
"""

import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.hybrid_nb import HybridNB

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("demo/artifacts")
CAT_THRESHOLD = 300
N_PER_CAT     = 25   # products to keep per category in the demo pool

DISPLAY_NAMES = {
    "Computers & Accessories":                  "Computers",
    "Headphones, Earbuds & Accessories":        "Headphones",
    "Camera & Photo":                           "Camera",
    "Television & Video":                       "TV & Video",
    "Portable Audio & Video":                   "Portable Audio",
    "Home Audio":                               "Home Audio",
    "Accessories & Supplies":                   "Accessories",
    "Car & Vehicle Electronics":                "Car Electronics",
    "GPS, Finders & Accessories":               "GPS",
    "Wearable Technology":                      "Wearables",
    "Security & Surveillance":                  "Security",
    "Electronics Warranties":                   "Other",
    "Power Accessories":                        "Power",
    "eBook Readers & Accessories":              "eReaders",
    "Household Batteries, Chargers & Accessories": "Batteries",
    "Video Projectors":                         "Projectors",
    "Unknown":                                  "Other",
    "Other":                                    "Other",
}


def get_fine_categories(train_df):
    counts = train_df["top_category"].value_counts()
    kept   = set(counts[counts >= CAT_THRESHOLD].index)
    return sorted(kept) + ["Other"]


def map_cat(raw, kept_set):
    return raw if raw in kept_set else "Other"


def build_product_pool(df, kept_set):
    products = (
        df.dropna(subset=["product_title"])
        .query("product_title.str.len() > 15 and price_known == 1 and average_rating >= 3.5",
               engine="python")
        .drop_duplicates("parent_asin")
        .copy()
    )
    products["fine_category"] = products["top_category"].apply(lambda x: map_cat(x, kept_set))
    products["display_category"] = products["fine_category"].map(DISPLAY_NAMES).fillna("Other")

    pool = (
        products
        .groupby("fine_category", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), N_PER_CAT), random_state=42))
        .reset_index(drop=True)
    )

    return pool[[
        "parent_asin", "product_title", "fine_category",
        "display_category", "price", "log_price", "average_rating",
    ]]


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    print(f"  {len(train):,} rows")

    categories = get_fine_categories(train)
    kept_set   = set(categories) - {"Other"}
    print(f"  {len(categories)} fine categories")

    train = train.copy()
    train["category"] = train["top_category"].apply(lambda x: map_cat(x, kept_set))

    print("\nFitting HybridNB...")
    model = HybridNB()
    model.fit(train, categories)
    print("  Done.")

    print("\nBuilding product pool...")
    pool = build_product_pool(train, kept_set)
    print(f"  {len(pool)} products across {pool['fine_category'].nunique()} categories")
    print(pool["display_category"].value_counts().to_string())

    print("\nSaving artifacts...")
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(ARTIFACTS_DIR / "categories.json", "w") as f:
        json.dump({
            "categories":    categories,
            "display_names": DISPLAY_NAMES,
        }, f, indent=2)

    pool.to_parquet(ARTIFACTS_DIR / "products.parquet", index=False)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("  model.pkl")
    print("  categories.json")
    print("  products.parquet")


if __name__ == "__main__":
    main()
