import numpy as np
from scipy.special import digamma


class DirichletMultinomial:
    """
    Dirichlet-Multinomial model for per-user category preferences.

    Generative story:
        theta_u ~ Dirichlet(alpha)
        category_i | theta_u ~ Multinomial(theta_u)   for each liked product

    Posterior after observing liked-category counts n_u:
        theta_u | n_u ~ Dirichlet(alpha + n_u)

    alpha is shared across all users and estimated by maximising the
    marginal likelihood via Minka's fixed-point iteration.
    """

    def __init__(self, max_iter=200, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_ = None
        self.categories_ = None

    def fit(self, user_counts):
        """
        user_counts: DataFrame of shape (n_users, n_categories).
        Each row is one user's liked-product counts per category.

        Fixed-point update (Minka 2000):
            alpha_k <- alpha_k * [sum_u psi(n_uk + alpha_k) - psi(alpha_k)]
                                / [sum_u psi(N_u  + alpha_0) - psi(alpha_0)]
        """
        N = user_counts.values.astype(float)
        row_totals = N.sum(axis=1)

        alpha = N.mean(axis=0) + 1e-3
        alpha = alpha / alpha.sum() * 10.0

        for _ in range(self.max_iter):
            alpha_0 = alpha.sum()
            num = (digamma(N + alpha) - digamma(alpha)).sum(axis=0)
            den = (digamma(row_totals[:, None] + alpha_0) - digamma(alpha_0)).sum()

            alpha_new = alpha * (num / den)
            alpha_new = np.clip(alpha_new, 1e-6, None)

            if np.max(np.abs(alpha_new - alpha)) < self.tol:
                break
            alpha = alpha_new

        self.alpha_ = alpha_new
        self.categories_ = list(user_counts.columns)
        return self

    def posterior(self, counts):
        return self.alpha_ + np.asarray(counts, dtype=float)

    def expected_theta(self, counts):
        alpha_post = self.posterior(counts)
        return alpha_post / alpha_post.sum()

    def mle_theta(self, counts):
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        if total == 0:
            return np.ones(len(counts)) / len(counts)
        return counts / total

    def recommend(self, counts, candidate_df, category_col="category", top_k=10, use_map=True):
        theta = self.expected_theta(counts) if use_map else self.mle_theta(counts)
        cat_to_score = dict(zip(self.categories_, theta))
        scores = candidate_df[category_col].map(cat_to_score).fillna(0.0)
        return (
            candidate_df.assign(rec_score=scores)
            .sort_values("rec_score", ascending=False)
            .head(top_k)
        )
