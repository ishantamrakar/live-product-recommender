import numpy as np
from scipy.stats import norm


class PerUserGaussianNB:
    """
    Per-user Gaussian NB over continuous product features (e.g. log_price).

    For each user we want P(like | feature_value, user). Using Bayes:

        P(like | x, u) ∝ P(x | like, u) * P(like | u)

    P(x | like, u) = N(x; mu_u_liked, sigma^2)

    mu_u_liked is estimated with a Gaussian prior centred on the population mean:

        MAP: mu_u_liked = (kappa_0 * mu_pop_liked + n_u * x_bar_u) / (kappa_0 + n_u)

    kappa_0 = 0  →  MLE (pure user mean, undefined if n_u = 0)
    kappa_0 > 0  →  MAP (shrink toward population mean)

    Variance is shared across users — too few per-user observations to estimate it reliably.
    """

    def __init__(self, kappa_0=1.0, feature="log_price"):
        self.kappa_0 = kappa_0
        self.feature = feature
        self.mu_pop_ = None     # [mu_not_liked, mu_liked] from population
        self.var_pop_ = None    # shared variance
        self.global_like_rate_ = None

    def fit(self, df):
        liked     = df.loc[df["liked"] == 1, self.feature].values
        not_liked = df.loc[df["liked"] == 0, self.feature].values

        self.mu_pop_ = np.array([not_liked.mean(), liked.mean()])
        self.var_pop_ = df[self.feature].var() + 1e-9
        self.global_like_rate_ = df["liked"].mean()
        return self

    def _posterior_mu(self, user_vals, class_idx):
        n = len(user_vals)
        if n == 0 or self.kappa_0 == 0:
            return self.mu_pop_[class_idx] if n == 0 else np.mean(user_vals)
        return (self.kappa_0 * self.mu_pop_[class_idx] + n * np.mean(user_vals)) / (self.kappa_0 + n)

    def score(self, x, user_liked_vals, user_not_liked_vals):
        """
        P(like | x) for one user given their history of liked/not-liked product prices.
        x can be a scalar or array of candidate product prices.
        """
        mu_liked     = self._posterior_mu(user_liked_vals,     class_idx=1)
        mu_not_liked = self._posterior_mu(user_not_liked_vals, class_idx=0)

        log_p_liked     = norm.logpdf(x, mu_liked,     np.sqrt(self.var_pop_))
        log_p_not_liked = norm.logpdf(x, mu_not_liked, np.sqrt(self.var_pop_))

        log_prior_liked     = np.log(self.global_like_rate_)
        log_prior_not_liked = np.log(1 - self.global_like_rate_)

        log_liked     = log_prior_liked     + log_p_liked
        log_not_liked = log_prior_not_liked + log_p_not_liked

        log_max = np.maximum(log_liked, log_not_liked)
        exp_liked     = np.exp(log_liked     - log_max)
        exp_not_liked = np.exp(log_not_liked - log_max)

        return exp_liked / (exp_liked + exp_not_liked)

    def predict(self, x, user_liked_vals, user_not_liked_vals, threshold=0.5):
        return (self.score(x, user_liked_vals, user_not_liked_vals) >= threshold).astype(int)
