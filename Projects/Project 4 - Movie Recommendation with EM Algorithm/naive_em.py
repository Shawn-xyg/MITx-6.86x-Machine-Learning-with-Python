"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]           # Get the data size n
    mu = mixture[0]          # Get the parameters: mu, var and p
    var = mixture[1]
    p = mixture[2]
    d = X.shape[1]           # Dimension of the dataset
    first_term = (2 * np.pi * var) ** (-d / 2)      # The denominator of normal distribution
    exponent = np.exp(np.linalg.norm(X[:, np.newaxis] - mu, axis = 2) ** 2 / -(2 * var))
    soft_counts = p * first_term * exponent         # The soft count (probability of each point belong to one Gaussian mixture model)
    total_counts = soft_counts.sum(axis=1).reshape(n, 1)
    weighted_soft_counts = np.divide(soft_counts, total_counts)     # Posterior probability
    loglike = np.sum(np.log(total_counts), axis=0)                  # Log-likelihood
    return weighted_soft_counts, float(loglike)



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    d = X.shape[1]
    k = post.shape[1]
    n_hat = np.sum(post, axis=0)  # Get the N_hat by adding up the posterior probability by column
    p_hat = n_hat / n
    mu_hat = (1 / n_hat.reshape(k, 1)) * post.T @ X
    norm = np.power(np.linalg.norm(X[:, np.newaxis] - mu_hat, axis=2), 2)
    summation = np.sum(post * norm, axis=0)
    var_hat = (1 / (n_hat * d)) * summation
    return GaussianMixture(mu_hat, var_hat, p_hat)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_likelihood = None
    likelihood = None
    while (prev_likelihood is None or likelihood - prev_likelihood >= 1e-6 * np.abs(likelihood)):
        prev_likelihood = likelihood
        post, likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, likelihood
    raise NotImplementedError
