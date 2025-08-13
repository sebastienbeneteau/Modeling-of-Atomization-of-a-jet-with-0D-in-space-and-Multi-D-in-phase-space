from scipy.stats import multivariate_normal
import numpy as np
import itertools

def double_factorial(n):
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result

def gaussian_moment(n, mu, sigma):
    moment = 0
    for k in range(n // 2 + 1):
        binom = np.math.comb(n, 2*k)
        double_fact = double_factorial(2*k - 1)
        term = binom * double_fact * sigma**(2*k) * mu**(n - 2*k)
        moment += term
    return moment

def gaussian_initial_moments(mu, sigma, N):
    return np.array([gaussian_moment(n, mu, sigma) for n in range(N)])



def MC_multinormal_moments(mu, cov, N, num_seeds = int(1e6)):
    """
    Generate a dictionnary of bivariate Gaussian moments using Monte Carlo method.
    """
    
    nodes = np.random.multivariate_normal(mu, cov, size=num_seeds)

    # Compute weights from the bivariate PDF
    pdf_values = multivariate_normal.pdf(nodes, mean=mu, cov=cov, allow_singular=True)
    weights = pdf_values / np.sum(pdf_values)
    
    moments = np.zeros(N)
    
    for idx in itertools.product(*[range(n) for n in N]):
        moments[idx] = np.sum(weights * np.prod([nodes[:,i] ** idx[i] for i in range(len(idx))], axis = 0))
    
    return moments, nodes, weights