import numpy as np


class BernoulliNB:
    """
    Bernoulli Naive Bayes from scratch.

    Each feature x_k is binary. The likelihood for one sample is:

        P(x | class=c) = prod_k P(x_k=1|c)^x_k * P(x_k=0|c)^(1-x_k)

    The key difference from Multinomial NB: the (1 - x_k) term means the
    absence of a feature is explicitly penalised, not ignored.

    MAP with a symmetric Beta(alpha, alpha) prior on each feature probability:

        P(x_k=1 | c) = (alpha + count(x_k=1, c)) / (2*alpha + n_c)

    alpha=0 recovers MLE. alpha=1 is Laplace smoothing toward 0.5.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None      # log P(x_k=1 | c)
        self.feature_log_prob_neg_ = None  # log P(x_k=0 | c)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))

        for k, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_counts[k]    = len(X_c)
            feature_counts[k]  = X_c.sum(axis=0)

        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        p = (feature_counts + self.alpha) / (class_counts[:, None] + 2 * self.alpha)
        self.feature_log_prob_     = np.log(p)
        self.feature_log_prob_neg_ = np.log(1 - p)
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_proba = (
            X @ self.feature_log_prob_.T
            + (1 - X) @ self.feature_log_prob_neg_.T
            + self.class_log_prior_
        )
        return log_proba

    def predict_proba(self, X):
        log_p = self.predict_log_proba(X)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
