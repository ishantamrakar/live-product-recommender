"""
Prior sensitivity sweep — how does alpha affect MNB performance?

Sweeps alpha across four orders of magnitude and records accuracy + F1
on the full val set, cold-start users, and established users separately.
Results saved to results/metrics/alpha_sweep.csv.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.multinomial_nb import MultinomialNB
from src.experiments.setup import load_data, build_vectoriser, RESULTS_DIR

ALPHAS = np.logspace(-3, 2, 40)  # 0.001 → 100, log-spaced


def score(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_not_liked": f1_score(y, preds, pos_label=0, zero_division=0),
        "f1_liked":     f1_score(y, preds, pos_label=1, zero_division=0),
    }


def main():
    print("Loading data...")
    train, val, _ = load_data()

    print("Vectorising...")
    vec, X_train, _ = build_vectoriser(train["review_text"])
    X_val = vec.transform(val["review_text"])

    y_train = train["liked"].values
    y_val   = val["liked"].values

    # Tag val users by how many reviews they have in training
    train_user_counts = train["user_id"].value_counts()
    val["n_train_reviews"] = val["user_id"].map(train_user_counts).fillna(0).astype(int)

    cold_mask  = val["n_train_reviews"] <= 3
    estab_mask = val["n_train_reviews"] > 10

    X_cold  = vec.transform(val.loc[cold_mask,  "review_text"])
    X_estab = vec.transform(val.loc[estab_mask, "review_text"])
    y_cold  = val.loc[cold_mask,  "liked"].values
    y_estab = val.loc[estab_mask, "liked"].values

    print(f"Val users — cold-start: {cold_mask.sum():,}  established: {estab_mask.sum():,}\n")
    print(f"Sweeping {len(ALPHAS)} alpha values from {ALPHAS[0]:.4f} to {ALPHAS[-1]:.1f}...\n")

    rows = []
    for alpha in ALPHAS:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)

        row = {"alpha": alpha}
        for split_name, X, y in [("val", X_val, y_val),
                                  ("cold", X_cold, y_cold),
                                  ("estab", X_estab, y_estab)]:
            s = score(model, X, y)
            row[f"{split_name}_accuracy"]     = s["accuracy"]
            row[f"{split_name}_f1_not_liked"] = s["f1_not_liked"]
            row[f"{split_name}_f1_liked"]     = s["f1_liked"]

        rows.append(row)

    results = pd.DataFrame(rows)

    out_path = RESULTS_DIR / "metrics" / "alpha_sweep.csv"
    results.to_csv(out_path, index=False)
    print(f"✓ Saved to {out_path}\n")

    # Quick summary at the terminal
    best_val   = results.loc[results["val_accuracy"].idxmax()]
    best_cold  = results.loc[results["cold_f1_not_liked"].idxmax()]
    best_estab = results.loc[results["estab_f1_not_liked"].idxmax()]

    print(f"{'Split':<12} {'Best alpha':>12}  {'Accuracy':>10}  {'F1(not-liked)':>14}")
    print("-" * 55)
    for label, row in [("val", best_val), ("cold-start", best_cold), ("established", best_estab)]:
        split = label.replace("-", "_").replace("established", "estab").replace("val", "val")
        key   = "cold" if "cold" in label else ("estab" if "estab" in label else "val")
        print(f"{label:<12} {row['alpha']:>12.4f}  {row[f'{key}_accuracy']:>10.4f}  {row[f'{key}_f1_not_liked']:>14.4f}")


if __name__ == "__main__":
    main()
