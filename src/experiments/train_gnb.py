"""
Per-user Gaussian NB: MLE vs MAP for price preference estimation.

For each user we build a personal Gaussian model over log_price:
  P(like | log_price, user) ∝ P(log_price | like, user) * P(like | user)

The class-conditional means are estimated from the user's own history,
with MAP shrinking toward the population mean when history is thin.

Evaluation: held-out val reviews for users who have training history.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.per_user_gaussian_nb import PerUserGaussianNB
from src.experiments.setup import PROCESSED_DIR, RESULTS_DIR


def build_user_history(df, feature="log_price"):
    liked     = df[df["liked"] == 1].groupby("user_id")[feature].apply(list)
    not_liked = df[df["liked"] == 0].groupby("user_id")[feature].apply(list)
    return liked, not_liked


def evaluate_per_user(model, val_df, train_liked, train_not_liked, feature="log_price"):
    y_true_mle, y_pred_mle = [], []
    y_true_map, y_pred_map = [], []

    for user_id, group in val_df.groupby("user_id"):
        liked_hist     = np.array(train_liked.get(user_id, []))
        not_liked_hist = np.array(train_not_liked.get(user_id, []))
        x = group[feature].values
        y = group["liked"].values

        mle = PerUserGaussianNB(kappa_0=0.0, feature=feature)
        mle.mu_pop_           = model.mu_pop_
        mle.var_pop_          = model.var_pop_
        mle.global_like_rate_ = model.global_like_rate_

        y_true_mle.extend(y)
        y_pred_mle.extend(mle.predict(x, liked_hist, not_liked_hist))
        y_true_map.extend(y)
        y_pred_map.extend(model.predict(x, liked_hist, not_liked_hist))

    return (
        np.array(y_true_mle), np.array(y_pred_mle),
        np.array(y_true_map), np.array(y_pred_map),
    )


def report(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"  {label:<30}  acc={acc:.3f}  F1(macro)={f1:.3f}")
    return f1


def main():
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

    print(f"  train reviews: {len(train):,}  |  val reviews: {len(val):,}")

    print("\nFitting population prior on train...")
    model = PerUserGaussianNB(kappa_0=5.0)
    model.fit(train)
    print(f"  mu_pop  liked={model.mu_pop_[1]:.4f}  not-liked={model.mu_pop_[0]:.4f}")
    print(f"  sigma_pop = {np.sqrt(model.var_pop_):.4f}")
    print(f"  global like rate = {model.global_like_rate_:.3f}")

    print("\nBuilding user histories from train...")
    train_liked, train_not_liked = build_user_history(train)

    val_users_with_history = val[val["user_id"].isin(train["user_id"])]
    print(f"  Val users with train history: {val_users_with_history['user_id'].nunique():,}")

    print("\nEvaluating...")
    y_true_mle, y_pred_mle, y_true_map, y_pred_map = evaluate_per_user(
        model, val_users_with_history, train_liked, train_not_liked
    )

    print("\n--- Overall ---")
    report(y_true_mle, y_pred_mle, "MLE")
    report(y_true_map, y_pred_map, f"MAP (kappa_0={model.kappa_0})")

    # Cold-start vs established based on train history length
    user_train_counts = train.groupby("user_id")["liked"].count()
    cold_users  = set(user_train_counts[user_train_counts <= 3].index)
    estab_users = set(user_train_counts[user_train_counts > 10].index)

    for label, users in [("Cold-start (<=3 train reviews)", cold_users),
                         ("Established (>10 train reviews)", estab_users)]:
        mask = val_users_with_history["user_id"].isin(users)
        sub  = val_users_with_history[mask]
        if len(sub) == 0:
            continue
        _, y_pred_mle_s, _, y_pred_map_s = evaluate_per_user(
            model, sub, train_liked, train_not_liked
        )
        y_true_s = sub["liked"].values
        print(f"\n--- {label} (n_reviews={len(sub):,}) ---")
        report(y_true_s, y_pred_mle_s, "MLE")
        report(y_true_s, y_pred_map_s, f"MAP (kappa_0={model.kappa_0})")

    # kappa_0 sweep
    print("\n--- kappa_0 sweep ---")
    kappas  = np.logspace(-1, 3, 30)
    results = []
    for k in kappas:
        m = PerUserGaussianNB(kappa_0=k)
        m.mu_pop_           = model.mu_pop_
        m.var_pop_          = model.var_pop_
        m.global_like_rate_ = model.global_like_rate_
        _, _, yt, yp = evaluate_per_user(m, val_users_with_history, train_liked, train_not_liked)
        f1 = f1_score(yt, yp, average="macro")
        results.append({"kappa_0": k, "val_f1": f1})

    sweep = pd.DataFrame(results)
    best  = sweep.loc[sweep["val_f1"].idxmax()]
    print(f"  Best kappa_0={best['kappa_0']:.3f}  val F1={best['val_f1']:.4f}")
    sweep.to_csv(RESULTS_DIR / "metrics" / "gnb_kappa_sweep.csv", index=False)
    print(f"\n  Saved sweep to results/metrics/gnb_kappa_sweep.csv")


if __name__ == "__main__":
    main()
