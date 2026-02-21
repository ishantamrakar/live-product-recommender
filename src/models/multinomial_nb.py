import numpy as np
from scipy.sparse import issparse


class MultinomialNB:
    """
    Multinomial Naive Bayes with support for MLE (alpha=0) and MAP (alpha>0).

    Under MLE the word likelihoods are pure frequency estimates:
        P(w | y) = count(w, y) / sum_w count(w, y)

    Under MAP we add a Dirichlet prior with concentration alpha, which
    reduces to Laplace smoothing when alpha=1:
        P(w | y) = (count(w, y) + alpha) / (sum_w count(w, y) + alpha * |V|)

    Working in log-space throughout to avoid floating-point underflow when
    multiplying many small probabilities across a long vocabulary.
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.classes_          = None
        self.class_log_prior_  = None
        self.feature_log_prob_ = None
        self.feature_count_    = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        self.feature_count_ = np.zeros((n_classes, n_features))
        class_counts        = np.zeros(n_classes)

        for k, c in enumerate(self.classes_):
            mask = (y == c)
            X_c  = X[mask]
            self.feature_count_[k] = X_c.sum(axis=0) if not issparse(X_c) else np.asarray(X_c.sum(axis=0)).ravel()
            class_counts[k]        = mask.sum()

        # MLE class prior: P(y) = count(y) / N
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())

        # Word likelihoods — alpha=0 is pure MLE, alpha>0 is MAP
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))

        return self

    def predict_log_proba(self, X):
        if issparse(X):
            log_likelihood = X @ self.feature_log_prob_.T
        else:
            log_likelihood = X @ self.feature_log_prob_.T

        log_posterior = self.class_log_prior_ + log_likelihood

        # Normalise with log-sum-exp for numerical stability
        log_z = np.log(np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        return log_posterior - log_posterior.max(axis=1, keepdims=True) - log_z

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def top_words(self, class_idx, vocab, n=10):
        """Words with highest P(word | class) — useful for inspection."""
        log_probs = self.feature_log_prob_[class_idx]
        top_idx   = np.argsort(log_probs)[-n:][::-1]
        return [(vocab[i], np.exp(log_probs[i])) for i in top_idx]
