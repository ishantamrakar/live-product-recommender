"""
Per-user Bernoulli NB: cross-category affinity model.

Features for each (user, product) pair:
  - 11 user history flags: liked_<category>_before (from train)
  - 11 product flags: is_<category> (one-hot of candidate product)

The user history flags capture which categories the user has engaged with.
The Bernoulli likelihood explicitly penalises absence — a user who has
never liked a Camera product gets a lower score for Camera candidates.

MAP: Beta(alpha, alpha) prior on each feature probability.
     alpha=0 -> MLE, alpha=1 -> Laplace smoothing toward 0.5.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.bernoulli_nb import BernoulliNB
from src.experiments.setup import PROCESSED_DIR, RESULTS_DIR

CATEGORIES = [
    "Computers", "Audio", "Camera", "TV", "Car",
    "GPS", "Wearables", "Security", "Power", "eReaders", "Other",
]


def build_features(df, user_history):
    """
    Build binary feature matrix for all rows in df.

    user_history: dict mapping user_id -> set of categories liked in train.
    """
    user_feat  = np.zeros((len(df), len(CATEGORIES)), dtype=float)
    prod_feat  = np.zeros((len(df), len(CATEGORIES)), dtype=float)
    cat_index  = {c: i for i, c in enumerate(CATEGORIES)}

    for i, (_, row) in enumerate(df.iterrows()):
        liked_cats = user_history.get(row["user_id"], set())
        for cat in liked_cats:
            if cat in cat_index:
                user_feat[i, cat_index[cat]] = 1.0
        prod_cat = row["category"]
        if prod_cat in cat_index:
            prod_feat[i, cat_index[prod_cat]] = 1.0

    return np.hstack([user_feat, prod_feat])


def build_user_history(df):
    return (
        df[df["liked"] == 1]
        .groupby("user_id")["category"]
        .apply(set)
        .to_dict()
    )


def evaluate(model, X, y, label):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="macro")
    print(f"\n{label}")
    print(f"  Accuracy: {acc:.4f}  F1(macro): {f1:.4f}")
    print(classification_report(y, preds, target_names=["not-liked", "liked"],
                                digits=3))
    return f1


def main():
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    print("Building user histories from train...")
    user_history = build_user_history(train)
    print(f"  Users with liked history: {len(user_history):,}")

    val_with_history = val[val["user_id"].isin(user_history)].reset_index(drop=True)
    print(f"  Val rows with user history: {len(val_with_history):,}")

    print("\nBuilding feature matrices...")
    print("  (train)")
    train_reset = train[train["user_id"].isin(user_history)].reset_index(drop=True)
    X_train = build_features(train_reset, user_history)
    y_train = train_reset["liked"].values

    print("  (val)")
    X_val = build_features(val_with_history, user_history)
    y_val = val_with_history["liked"].values

    print(f"  Feature matrix shape: {X_train.shape}  "
          f"(11 user history + 11 product category)")

    print("\n--- MLE (alpha=0) ---")
    mle = BernoulliNB(alpha=0.0).fit(X_train, y_train)
    evaluate(mle, X_val, y_val, "Val")

    print("\n--- MAP (alpha=1, Laplace) ---")
    lap = BernoulliNB(alpha=1.0).fit(X_train, y_train)
    evaluate(lap, X_val, y_val, "Val")

    # Cold-start vs established
    user_train_counts = train.groupby("user_id")["liked"].count()
    cold_users  = set(user_train_counts[user_train_counts <= 3].index)
    estab_users = set(user_train_counts[user_train_counts > 10].index)

    for label, users in [("Cold-start (<=3 liked in train)", cold_users),
                         ("Established (>10 liked in train)", estab_users)]:
        mask = val_with_history["user_id"].isin(users)
        if mask.sum() == 0:
            continue
        print(f"\n--- {label} (n={mask.sum():,}) ---")
        evaluate(mle, X_val[mask], y_val[mask], "MLE")
        evaluate(lap, X_val[mask], y_val[mask], "MAP (alpha=1)")

    # alpha sweep
    print("\n--- alpha sweep ---")
    alphas  = np.logspace(-2, 2, 30)
    results = []
    for a in alphas:
        m  = BernoulliNB(alpha=a).fit(X_train, y_train)
        f1 = f1_score(y_val, m.predict(X_val), average="macro")
        results.append({"alpha": a, "val_f1": f1})

    sweep = pd.DataFrame(results)
    best  = sweep.loc[sweep["val_f1"].idxmax()]
    print(f"  Best alpha={best['alpha']:.4f}  val F1={best['val_f1']:.4f}")

    sweep.to_csv(RESULTS_DIR / "metrics" / "bnb_alpha_sweep.csv", index=False)
    print("  Saved to results/metrics/bnb_alpha_sweep.csv")

    # Cross-category affinity: what does the model learn?
    print("\n--- Learned P(liked_<category>_before | liked=1) [top 5] ---")
    feature_names = [f"user_liked_{c}" for c in CATEGORIES] + [f"is_{c}" for c in CATEGORIES]
    p_given_liked = np.exp(lap.feature_log_prob_[1])
    top5 = np.argsort(p_given_liked)[-5:][::-1]
    for i in top5:
        print(f"  {feature_names[i]:<30}  P(feat=1|liked)={p_given_liked[i]:.3f}")


if __name__ == "__main__":
    main()
