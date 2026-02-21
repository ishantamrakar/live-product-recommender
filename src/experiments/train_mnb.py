import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.multinomial_nb import MultinomialNB

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_splits():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    return train, val


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, pos_label=0)  # F1 on minority class (not-liked)
    print(f"{label:10s}  accuracy={acc:.4f}  F1(not-liked)={f1:.4f}")
    return acc, f1


def main():
    print("Loading data...")
    train, val = load_splits()

    # Use raw word counts — this is pure MLE territory.
    # TF-IDF would bake in a discriminative weighting that doesn't correspond
    # to the generative MLE story we're telling.
    print("Vectorising text...")
    vectoriser = CountVectorizer(max_features=20_000, min_df=2, ngram_range=(1, 2),
                                stop_words="english")
    X_train = vectoriser.fit_transform(train["review_text"])
    X_val   = vectoriser.transform(val["review_text"])

    y_train = train["liked"].values
    y_val   = val["liked"].values

    vocab = np.array(vectoriser.get_feature_names_out())
    print(f"Vocabulary size: {len(vocab):,}\n")

    # --- MLE (alpha=0) ---
    print("=" * 60)
    print("MLE  (alpha=0) — no smoothing")
    print("=" * 60)
    mle = MultinomialNB(alpha=0.0)
    mle.fit(X_train, y_train)
    evaluate(mle, X_train, y_train, "train")
    evaluate(mle, X_val,   y_val,   "val")

    print("\nTop words for LIKED class:")
    for word, prob in mle.top_words(1, vocab, n=10):
        print(f"  {word:<25} {prob:.6f}")

    print("\nTop words for NOT-LIKED class:")
    for word, prob in mle.top_words(0, vocab, n=10):
        print(f"  {word:<25} {prob:.6f}")

    # Log-odds ratio: log P(w|liked) - log P(w|not-liked)
    # High positive = strongly predicts liked, high negative = strongly predicts not-liked
    print("\nMost discriminative words (log-odds ratio):")
    log_odds = mle.feature_log_prob_[1] - mle.feature_log_prob_[0]
    for sign, label in [(1, "→ LIKED"), (-1, "→ NOT-LIKED")]:
        idx = np.argsort(sign * log_odds)[-10:][::-1]
        print(f"\n  {label}:")
        for i in idx:
            print(f"    {vocab[i]:<30} {log_odds[i]:+.3f}")

    print("\n--- Zero-frequency problem ---")
    for k, label in enumerate(["not-liked", "liked"]):
        zeros = (mle.feature_count_[k] == 0).sum()
        print(f"Words with zero count in {label:10s}: {zeros:,} / {len(vocab):,}")
    print("\nThese cause log P(w|class) = -inf, making log-odds = ±inf")
    print("One unseen word is enough to send the posterior to 0 or 1 with certainty")

    # --- MAP (alpha=1) for comparison ---
    print("\n" + "=" * 60)
    print("MAP  (alpha=1) — Laplace smoothing")
    print("=" * 60)
    map_model = MultinomialNB(alpha=1.0)
    map_model.fit(X_train, y_train)
    evaluate(map_model, X_train, y_train, "train")
    evaluate(map_model, X_val,   y_val,   "val")

    zero_features_map = (np.exp(map_model.feature_log_prob_[1]) == 0).sum()
    print(f"\nWords with P(w | liked)=0 after smoothing: {zero_features_map:,}")

    # --- Cold-start vs established users ---
    # The prior only matters when data is scarce. Users who appear rarely in
    # the training set have niche review vocabulary — more likely to hit
    # zero-probability words under MLE. MAP should help them the most.
    print("\n" + "=" * 60)
    print("Cold-start vs established users")
    print("=" * 60)

    train_user_counts = train["user_id"].value_counts()
    val["user_review_count"] = val["user_id"].map(train_user_counts).fillna(0).astype(int)

    groups = {
        "cold-start  (≤3 reviews)" : val["user_review_count"] <= 3,
        "established (>10 reviews)": val["user_review_count"] > 10,
    }

    print(f"\n{'Group':<30} {'N':>6}  {'MLE acc':>8}  {'MAP acc':>8}  {'MLE F1':>8}  {'MAP F1':>8}")
    print("-" * 75)

    for group_label, mask in groups.items():
        subset = val[mask]
        if len(subset) == 0:
            continue

        X_sub = vectoriser.transform(subset["review_text"])
        y_sub = subset["liked"].values

        mle_preds = mle.predict(X_sub)
        map_preds = map_model.predict(X_sub)

        mle_acc = accuracy_score(y_sub, mle_preds)
        map_acc = accuracy_score(y_sub, map_preds)
        mle_f1  = f1_score(y_sub, mle_preds, pos_label=0, zero_division=0)
        map_f1  = f1_score(y_sub, map_preds, pos_label=0, zero_division=0)

        print(f"{group_label:<30} {len(subset):>6}  {mle_acc:>8.4f}  {map_acc:>8.4f}  {mle_f1:>8.4f}  {map_f1:>8.4f}")


if __name__ == "__main__":
    main()
