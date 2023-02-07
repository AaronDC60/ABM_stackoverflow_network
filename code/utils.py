"""Module containing helper functions used to simulate the model."""

# Imports
import numpy as np
import scipy.stats

def draw_normal(mu, sigma):
        """
        Draw probability from normal distribution.

        Parameters
        ----------
        mu : float
            mean of the distribution
        sigma : float
            std of the distribution

        Returns
        -------
        p : float
            probability
        """
        # value bounded between 0 and 1
        lower = 0
        upper = 1

        return scipy.stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)

def draw_uniform():
    """
    Draw probability from uniform distribution.

    Returns
    -------
    p : float
        probability
    """
    return np.random.uniform()

def draw_exponential(alpha):
    """
    Draw probability from exponential distribution.

    Parameters
    ----------
    alpha : int
        rate

    Returns
    -------
    p : float
        probability
    """
    return scipy.stats.truncexpon.rvs(alpha)/alpha

def calc_pdf(array):
    """
    Calculate the probability density function of an array.

    Parameters
    ----------
    array : numpy.ndarray
        list with outcomes

    Returns
    -------
    pdf : numpy.ndarray
        probability density function
    """
    pdf = np.zeros(np.max(array) + 1)
    for element in array:
        pdf[element] += 1

    return pdf / np.sum(pdf)

def calc_cdf(pdf):
    """
    Calculate the cummulative distribution function of a given pdf.

    Parameters
    ----------
    pdf : numpy.ndarray
        probability density function

    Returns
    -------
    cdf : numpy.ndarray
        cummulative distribution function
    """
    cdf = []
    value = 0

    for p in pdf:
        value += p
        cdf.append(value)

    return np.array(cdf)
