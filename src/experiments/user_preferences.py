"""
MLE vs MAP for per-user category preference estimation.

For each user in the val set we compute:
  - MLE estimate: liked_count_k / total_likes  (pure frequency)
  - MAP estimate: (alpha_k + liked_count_k) / (alpha_0 + total_likes)

Then we evaluate how well each estimate predicts the user's held-out
liked categories on the test set.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.dirichlet_multinomial import DirichletMultinomial
from src.experiments.setup import PROCESSED_DIR, RESULTS_DIR

CATEGORIES = [
    "Computers", "Audio", "Camera", "TV", "Car",
    "GPS", "Wearables", "Security", "Power", "eReaders", "Other",
]


def build_user_counts(df):
    liked = df[df["liked"] == 1]
    counts = (
        liked.groupby("user_id")["category"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORIES, fill_value=0)
    )
    return counts


def log_loss_dirichlet(alpha, counts_matrix):
    """
    Average per-user log-likelihood under Dirichlet-Multinomial.
    Higher = better fit.
    """
    from scipy.special import gammaln
    N = counts_matrix.values.astype(float)
    alpha_0 = alpha.sum()
    row_totals = N.sum(axis=1)

    ll = (
        gammaln(alpha_0) - gammaln(row_totals + alpha_0)
        + (gammaln(N + alpha) - gammaln(alpha)).sum(axis=1)
    )
    return ll.mean()


def top1_accuracy(theta, test_counts_row):
    pred = np.argmax(theta)
    actual_top = np.argmax(test_counts_row)
    return int(pred == actual_top)


def main():
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    print("Building user-category count matrices...")
    train_counts = build_user_counts(train)
    val_counts   = build_user_counts(val)
    test_counts  = build_user_counts(test)

    print(f"  Users in train: {len(train_counts):,}")
    print(f"  Users in val  : {len(val_counts):,}")

    print("\nFitting Dirichlet prior on training users...")
    model = DirichletMultinomial()
    model.fit(train_counts)

    print("\nLearned alpha (population prior):")
    for cat, a in sorted(zip(CATEGORIES, model.alpha_), key=lambda x: -x[1]):
        print(f"  {cat:<12}  alpha={a:.3f}")

    print(f"\n  alpha_0 = {model.alpha_.sum():.3f}  "
          f"(effective sample size of the prior)")

    # --- Evaluate on val users who also appear in test ---
    shared_users = val_counts.index.intersection(test_counts.index)
    print(f"\nUsers with val AND test likes: {len(shared_users):,}")

    val_counts_s  = val_counts.loc[shared_users]
    test_counts_s = test_counts.loc[shared_users]

    val_totals = val_counts_s.sum(axis=1)
    cold_mask  = val_totals <= 3
    estab_mask = val_totals > 10

    results = []
    for user in shared_users:
        counts = val_counts_s.loc[user].values
        test_c = test_counts_s.loc[user].values

        if test_c.sum() == 0:
            continue

        theta_mle = model.mle_theta(counts)
        theta_map = model.expected_theta(counts)

        results.append({
            "user_id":    user,
            "n_val":      counts.sum(),
            "mle_top1":   top1_accuracy(theta_mle, test_c),
            "map_top1":   top1_accuracy(theta_map, test_c),
        })

    results = pd.DataFrame(results)

    def report(mask, label):
        sub = results[mask]
        if len(sub) == 0:
            return
        print(f"\n{label}  (n={len(sub):,})")
        print(f"  MLE top-1 accuracy: {sub['mle_top1'].mean():.3f}")
        print(f"  MAP top-1 accuracy: {sub['map_top1'].mean():.3f}")
        print(f"  MAP lift           : {sub['map_top1'].mean() - sub['mle_top1'].mean():+.3f}")

    print("\n" + "=" * 50)
    print("Top-1 category prediction accuracy")
    print("=" * 50)
    report(results["user_id"].isin(val_counts_s.index[cold_mask]),  "Cold-start (≤3 likes in val)")
    report(results["user_id"].isin(val_counts_s.index[estab_mask]), "Established (>10 likes in val)")
    report([True] * len(results),                                    "All users")

    out = RESULTS_DIR / "metrics" / "user_preferences.csv"
    results.to_csv(out, index=False)
    print(f"\n✓ Saved to {out}")


if __name__ == "__main__":
    main()
