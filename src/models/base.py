"""
Base class for Bayesian models.

This module defines the abstract interface that all Bayesian models must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseBayesianModel(ABC):
    """
    Abstract base class for all Bayesian models.

    All concrete model implementations (Multinomial NB, Gaussian NB, etc.)
    should inherit from this class and implement the required methods.
    """

    def __init__(self, alpha: float = 1.0, prior_type: str = 'map'):
        """
        Initialize the Bayesian model.

        Parameters
        ----------
        alpha : float, default=1.0
            Prior hyperparameter for smoothing.
            - alpha = 0: Maximum Likelihood Estimation (MLE)
            - alpha = 1: Laplace smoothing (uniform prior)
            - alpha > 1: Stronger prior
        prior_type : str, default='map'
            Type of prior estimation to use.
            - 'mle': Maximum Likelihood Estimation
            - 'map': Maximum A Posteriori with fixed alpha
            - 'empirical': Empirical Bayes (learn alpha from data)
        """
        self.alpha = alpha
        self.prior_type = prior_type
        self.is_fitted = False

        # These will be set during training
        self.classes_ = None
        self.class_prior_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseBayesianModel':
        """
        Fit the Bayesian model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target labels

        Returns
        -------
        self : BaseBayesianModel
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        pass

    @abstractmethod
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log class probabilities for samples in X.

        Useful for numerical stability and for understanding
        the contribution of individual features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        log_proba : np.ndarray of shape (n_samples, n_classes)
            Log class probabilities for each sample
        """
        pass

    def _compute_class_prior(self, y: np.ndarray) -> np.ndarray:
        """
        Compute class prior probabilities with smoothing.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target labels

        Returns
        -------
        class_prior : np.ndarray of shape (n_classes,)
            Prior probability for each class
        """
        n_samples = len(y)
        n_classes = len(self.classes_)

        # Count samples per class
        class_counts = np.array([np.sum(y == c) for c in self.classes_])

        if self.prior_type == 'mle':
            # Maximum Likelihood: P(y) = count(y) / N
            class_prior = class_counts / n_samples
        else:  # MAP or empirical
            # MAP with Dirichlet prior: P(y) = (count(y) + alpha) / (N + alpha * K)
            class_prior = (class_counts + self.alpha) / (n_samples + self.alpha * n_classes)

        return class_prior

    def _optimize_alpha(self, X: np.ndarray, y: np.ndarray,
                       alpha_range: tuple = (0.01, 100)) -> float:
        """
        Optimize the alpha hyperparameter using cross-validation.

        This implements Empirical Bayes by finding the alpha that
        maximizes the marginal likelihood on held-out data.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target labels
        alpha_range : tuple, default=(0.01, 100)
            Range of alpha values to search

        Returns
        -------
        best_alpha : float
            Optimized alpha value
        """
        # TODO: Implement grid search or optimize.minimize_scalar
        # For now, return default alpha
        return self.alpha

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns
        -------
        importance : dict
            Feature importance scores (implementation-dependent)
        """
        raise NotImplementedError("Subclasses should implement feature importance")

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns
        -------
        params : dict
            Model parameters and hyperparameters
        """
        return {
            'alpha': self.alpha,
            'prior_type': self.prior_type,
            'is_fitted': self.is_fitted,
            'classes': self.classes_,
            'class_prior': self.class_prior_
        }
