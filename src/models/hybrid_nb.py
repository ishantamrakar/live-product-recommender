import numpy as np
from scipy.stats import norm
from scipy.special import digamma


RATING_EDGES = [3.5, 4.0, 4.5]   # produces 4 bins


def rating_bin_features(avg_rating):
    """One-hot vector of length 4 for the rating bin."""
    feat = np.zeros(4)
    feat[np.searchsorted(RATING_EDGES, avg_rating)] = 1.0
    return feat


class HybridNB:
    """
    Product-of-experts combination of Dirichlet-Multinomial and Bernoulli NB.

    log score(user, product) =
        log P(liked | user)                    [per-user like rate prior]
      + w_dir  * log P(category | user, liked) [Dirichlet-Multinomial]
      + w_bern * log P(features | liked)       [Bernoulli: cross-category + quality]

    Per-user like rate uses a Beta-Binomial MAP estimate:
        rate_map = (alpha_0 * global_rate + n_liked) / (alpha_0 + n_total)

    Bernoulli features (per product/user pair):
        - 11 user history flags: liked_<category>_before
        - 11 product category flags: is_<category>
        - 4  rating bin flags: rating_bin_0..3
    """

    def __init__(self, w_dir=1.0, w_bern=1.0, like_rate_alpha=10.0):
        self.w_dir   = w_dir
        self.w_bern  = w_bern
        self.like_rate_alpha = like_rate_alpha  # Beta-Binomial prior strength

        self.categories_     = None
        self.alpha_liked_    = None
        self.alpha_notliked_ = None

        self.bern_log_prob_liked_        = None
        self.bern_log_prob_notliked_     = None
        self.bern_log_prob_neg_liked_    = None
        self.bern_log_prob_neg_notliked_ = None

        self.global_like_rate_ = None

    def _fit_dirichlet(self, user_counts, max_iter=200, tol=1e-6):
        N = user_counts.values.astype(float)
        row_totals = N.sum(axis=1)
        mask = row_totals > 0
        N, row_totals = N[mask], row_totals[mask]

        alpha = N.mean(axis=0) + 1e-3
        alpha = alpha / alpha.sum() * 10.0

        for _ in range(max_iter):
            alpha_0  = alpha.sum()
            num = (digamma(N + alpha) - digamma(alpha)).sum(axis=0)
            den = (digamma(row_totals[:, None] + alpha_0) - digamma(alpha_0)).sum()
            alpha_new = np.clip(alpha * (num / den), 1e-6, None)
            if np.max(np.abs(alpha_new - alpha)) < tol:
                break
            alpha = alpha_new

        return alpha_new

    def fit(self, train_df, categories, bern_alpha=1.0):
        self.categories_       = categories
        self.global_like_rate_ = train_df["liked"].mean()
        cat_index = {c: i for i, c in enumerate(categories)}
        n_cats    = len(categories)
        n_feat    = n_cats + n_cats + 4     # user_hist + prod_cat + rating_bins

        def user_counts(df):
            return (
                df.groupby("user_id")["category"]
                .value_counts()
                .unstack(fill_value=0)
                .reindex(columns=categories, fill_value=0)
            )

        self.alpha_liked_    = self._fit_dirichlet(user_counts(train_df[train_df["liked"] == 1]))
        self.alpha_notliked_ = self._fit_dirichlet(user_counts(train_df[train_df["liked"] == 0]))

        # Build Bernoulli feature matrix vectorised
        user_history = (
            train_df[train_df["liked"] == 1]
            .groupby("user_id")["category"]
            .apply(set)
            .to_dict()
        )

        counts_liked    = np.zeros(n_feat)
        counts_notliked = np.zeros(n_feat)
        n_liked, n_notliked = 0, 0

        for _, row in train_df.iterrows():
            hist = user_history.get(row["user_id"], set())
            x = self._build_bern_features(
                row["category"], row["average_rating"], hist, cat_index, n_cats
            )
            if row["liked"] == 1:
                counts_liked += x
                n_liked += 1
            else:
                counts_notliked += x
                n_notliked += 1

        p_liked    = (counts_liked    + bern_alpha) / (n_liked    + 2 * bern_alpha)
        p_notliked = (counts_notliked + bern_alpha) / (n_notliked + 2 * bern_alpha)

        self.bern_log_prob_liked_        = np.log(p_liked)
        self.bern_log_prob_notliked_     = np.log(p_notliked)
        self.bern_log_prob_neg_liked_    = np.log(1 - p_liked)
        self.bern_log_prob_neg_notliked_ = np.log(1 - p_notliked)

        return self

    def _build_bern_features(self, category, avg_rating, user_history_set, cat_index, n_cats):
        user_feat = np.array([1.0 if c in user_history_set else 0.0
                              for c in self.categories_])
        prod_feat = np.zeros(n_cats)
        if category in cat_index:
            prod_feat[cat_index[category]] = 1.0
        return np.concatenate([user_feat, prod_feat, rating_bin_features(avg_rating)])

    def _user_log_prior(self, n_liked, n_total):
        r   = self.global_like_rate_
        a0  = self.like_rate_alpha
        rate = (a0 * r + n_liked) / (a0 + n_total)
        rate = np.clip(rate, 1e-9, 1 - 1e-9)
        return np.log(rate) - np.log(1 - rate)

    def _dirichlet_log_ratio(self, category, user_liked_counts, user_notliked_counts):
        if category not in self.categories_:
            return 0.0
        idx = self.categories_.index(category)

        alpha_post_liked    = self.alpha_liked_    + np.asarray(user_liked_counts)
        alpha_post_notliked = self.alpha_notliked_ + np.asarray(user_notliked_counts)

        return (
            np.log(alpha_post_liked[idx])    - np.log(alpha_post_liked.sum())
            - np.log(alpha_post_notliked[idx]) + np.log(alpha_post_notliked.sum())
        )

    def _bernoulli_log_ratio(self, category, avg_rating, user_history_set):
        cat_index = {c: i for i, c in enumerate(self.categories_)}
        x = self._build_bern_features(
            category, avg_rating, user_history_set, cat_index, len(self.categories_)
        )
        log_liked    = (x * self.bern_log_prob_liked_    + (1 - x) * self.bern_log_prob_neg_liked_).sum()
        log_notliked = (x * self.bern_log_prob_notliked_ + (1 - x) * self.bern_log_prob_neg_notliked_).sum()
        return log_liked - log_notliked

    def log_odds(self, category, avg_rating,
                 user_liked_counts, user_notliked_counts,
                 user_history_set,
                 user_n_liked, user_n_total):
        return (
            self._user_log_prior(user_n_liked, user_n_total)
            + self.w_dir  * self._dirichlet_log_ratio(category, user_liked_counts, user_notliked_counts)
            + self.w_bern * self._bernoulli_log_ratio(category, avg_rating, user_history_set)
        )
