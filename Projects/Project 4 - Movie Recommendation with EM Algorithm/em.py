"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    mu, var, p = mixture  # Unpacking the mixture
    sparse_matrix = np.where(X > 0, 1, 0)  # Creating a matrix recording the dimension of the original matrix, each cell equals 1 when positive and 0 when negative
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))  # Sum the sparse matrix based on column to get the dimensionality of each data point with data in it
    first_term = -d / 2 * np.log(2 * np.pi * var)  # Denominator of normal distribution
    exponent = np.linalg.norm((X[:, np.newaxis] - mu) * sparse_matrix[:, np.newaxis], axis=2) ** 2 / -(2 * var)  # Vectorized Exponential term of normal distribution
    log_p = np.log(p + 1e-16)  # log-transform the terms to prevent numerical underflow
    log_norm = first_term + exponent  # log-transform the terms to prevent numerial underflow
    f_uj = log_p + log_norm  # Redefine the log-transformed terms as a function f_uj
    log_post = f_uj - logsumexp(f_uj, axis=1).reshape((n, 1))  # The weighted soft count in log form
    origin_post = np.exp(log_post)  # Exponential the log_post variable to get the original soft count of each data point
    log_likelihood = np.sum(logsumexp(f_uj, axis=1).reshape((n, 1)), axis=0)  # The log likelihood would then be the sum of all the logsumexp term (the total count in log form)
    return origin_post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu, var, p = mixture  # Unpack the mixture components
    n = X.shape[0]  # Dimension of X dataset
    sparse_matrix = np.where(X > 0, 1,
                             0)  # Creating a sparse matrix indicating the position of filled points (matrix Cu)
    d = np.sum(sparse_matrix, axis=1).reshape((n, 1))  # The dimension of Cu for each point
    n_hat = np.sum(post, axis=0)  # Posterior probability for calculating the new probability p_hat
    log_p_hat = np.log(n_hat) - np.log(n)  # The new_probability p_hat under log domain to prevent numerical underflow
    p_hat = np.exp(log_p_hat)  # The true probability p_hat
    mu_denominator = post.T @ sparse_matrix  # Denominator for calculating mu_hat p(j|u) * delta(i, Cu)
    mu_update_indicator_matrix = np.where(mu_denominator >= 1, 1,
                                          0)  # Whether the denominator is larger than or equal to 1 to determine to perform update or not
    mu_hat = np.where(np.divide(post.T @ X, mu_denominator, out=np.zeros_like(post.T @ X),
                                where=mu_denominator != 0) * mu_update_indicator_matrix == 0, mu,
                      np.divide(post.T @ X, mu_denominator, out=np.zeros_like(post.T @ X),
                                where=mu_denominator != 0))  # Perform the update when the post.T @ X is not equal to 0, otherwise not updating
    var_denominator = np.sum(d * post, axis=0)  # The logsumexp of the denominator of variance factor
    var_nominator = np.sum(
        post * np.power(np.linalg.norm((X[:, np.newaxis] - mu_hat) * sparse_matrix[:, np.newaxis], axis=2), 2),
        axis=0)  # The logsumexp of the numerator of variance factor
    var_hat = np.divide(var_nominator, var_denominator, out=np.zeros_like(var_nominator),
                        where=var_denominator != 0)  # The variance under the log domain
    var_hat_withmin = np.where(var_hat > min_variance, var_hat,
                               min_variance)  # True variance regulated by the min_variance argument
    return GaussianMixture(mu_hat, var_hat_withmin, p_hat)
    raise NotImplementedError


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
        mixture = mstep(X, post, mixture)
    return mixture, post, likelihood
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu, var, p = mixture
    post, likelihood = estep(X, mixture)
    update_indicator_matrix = np.where(X != 0, 1, 0)
    predicted_value = post @ mu
    X_pred = np.where(update_indicator_matrix * X == 0, predicted_value, X)
    return X_pred
