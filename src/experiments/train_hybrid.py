"""
Ablation study: Dirichlet vs Bernoulli vs hybrid combinations.
Now includes per-user like rate prior and product quality (rating bin) features.

Metric: AUC.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hybrid_nb import HybridNB
from src.experiments.setup import PROCESSED_DIR, RESULTS_DIR

CAT_FREQ_THRESHOLD = 300
CAT_COLUMN = "top_category"


def get_categories(train_df):
    counts = train_df[CAT_COLUMN].value_counts()
    kept   = set(counts[counts >= CAT_FREQ_THRESHOLD].index)
    return sorted(kept) + ["Other"]


def map_category(raw, kept):
    return raw if raw in kept else "Other"


def build_user_history(df, categories):
    liked     = df[df["liked"] == 1].groupby("user_id")
    not_liked = df[df["liked"] == 0].groupby("user_id")

    def cat_counts(g):
        return (
            g["category"]
            .value_counts()
            .reindex(categories, fill_value=0)
            .values
        )

    return {
        "liked_counts":    {uid: cat_counts(g) for uid, g in liked},
        "notliked_counts": {uid: cat_counts(g) for uid, g in not_liked},
        "liked_sets":      {uid: set(g["category"].values) for uid, g in liked},
        "n_liked":         df[df["liked"] == 1].groupby("user_id").size().to_dict(),
        "n_total":         df.groupby("user_id").size().to_dict(),
    }


def score_val(model, val_df, history, weights, categories):
    w_dir, w_bern = weights
    model.w_dir  = w_dir
    model.w_bern = w_bern

    zeros = np.zeros(len(categories))
    scores, labels = [], []

    for _, row in val_df.iterrows():
        uid = row["user_id"]
        s = model.log_odds(
            category            = row["category"],
            avg_rating          = row["average_rating"],
            user_liked_counts   = history["liked_counts"].get(uid, zeros),
            user_notliked_counts= history["notliked_counts"].get(uid, zeros),
            user_history_set    = history["liked_sets"].get(uid, set()),
            user_n_liked        = history["n_liked"].get(uid, 0),
            user_n_total        = history["n_total"].get(uid, 0),
        )
        scores.append(s)
        labels.append(row["liked"])

    return roc_auc_score(labels, scores)


def main():
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    CATEGORIES = get_categories(train)
    kept_set   = set(CATEGORIES) - {"Other"}
    print(f"  Fine categories: {len(CATEGORIES)}  (threshold >= {CAT_FREQ_THRESHOLD})")

    train = train.copy()
    val   = val.copy()
    train[CAT_COLUMN] = train[CAT_COLUMN].apply(lambda x: map_category(x, kept_set))
    val[CAT_COLUMN]   = val[CAT_COLUMN].apply(lambda x: map_category(x, kept_set))

    # Use CAT_COLUMN as the working category column
    train["category"] = train[CAT_COLUMN]
    val["category"]   = val[CAT_COLUMN]

    val_with_history = val[val["user_id"].isin(train["user_id"])].reset_index(drop=True)
    print(f"  Val rows with train history: {len(val_with_history):,}")

    print("\nFitting model on train (Dirichlet + Bernoulli)...")
    model = HybridNB()
    model.fit(train, CATEGORIES)
    print("  Done.")

    print("\nBuilding user histories...")
    history = build_user_history(train, CATEGORIES)

    user_train_counts = train.groupby("user_id")["liked"].count()
    cold_users  = set(user_train_counts[user_train_counts <= 3].index)
    estab_users = set(user_train_counts[user_train_counts >  10].index)

    cold_val  = val_with_history[val_with_history["user_id"].isin(cold_users)]
    estab_val = val_with_history[val_with_history["user_id"].isin(estab_users)]

    configs = [
        ("Dirichlet only",          (1, 0)),
        ("Bernoulli only",          (0, 1)),
        ("Dirichlet + Bernoulli",   (1, 1)),
    ]

    results = []
    print("\n{:<28}  {:>8}  {:>12}  {:>12}".format(
        "Model", "All AUC", "Cold AUC", "Estab AUC"))
    print("-" * 66)

    for name, weights in configs:
        auc_all   = score_val(model, val_with_history, history, weights, CATEGORIES)
        auc_cold  = score_val(model, cold_val,  history, weights, CATEGORIES) if len(cold_val)  > 0 else float("nan")
        auc_estab = score_val(model, estab_val, history, weights, CATEGORIES) if len(estab_val) > 0 else float("nan")

        print(f"{name:<28}  {auc_all:>8.4f}  {auc_cold:>12.4f}  {auc_estab:>12.4f}")
        results.append({"model": name, "auc_all": auc_all,
                        "auc_cold": auc_cold, "auc_estab": auc_estab})

    out = RESULTS_DIR / "metrics" / "hybrid_ablation.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
