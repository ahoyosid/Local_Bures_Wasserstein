"""Gaussian transport utils

Author: XXX
2019
"""
import numpy as np
import torch
from torch import nn


def root_p_matrix(X, p=0.5):
    """ Computes the p-root of a matrix X.
    """
    U, S, V = torch.svd(X)
    S_root = S.clamp(min=0).pow(p)
    return U.mm(torch.diag(S_root)).mm(V.t())


def geometric_mean(X, Y):
    X_root = root_p_matrix(X, p=0.5)
    Y_root = root_p_matrix(Y, p=0.5)
    X_inv_root = root_p_matrix(X, p=-0.5)

    G = X_inv_root.matmul(Y.matmul(X_inv_root))
    G_sqrt = root_p_matrix(G, p=0.5)
    M = X_root.matmul(G_sqrt.matmul(X_root))
    return M


def plan_gaussian_transport(covariance_1, covariance_2, t=0.5):
    """ Computes the Bures-Wasserstein transport plan:
        T: N(0, cov1) -> N(0, cov2)

    Parameters
    ----------
    cov1: covariance of the source Gaussian.
        torch tensor shape=(n_components, n_components)

    cov2: covariance of the target Gaussian.
        torch tensor shape=(n_components, n_components)

    Returns
    -------
    M: transport plan.
        torch tensor shape=(n_components, n_components)
    """
    X_root = root_p_matrix(covariance_1, p=0.5)
    X_inv_root = root_p_matrix(covariance_1, p=-0.5)

    G = X_root.matmul(covariance_2.matmul(X_root))
    G_sqrt = root_p_matrix(G, p=t)
    M = X_inv_root.matmul(G_sqrt.matmul(X_inv_root))    
    return M


def l2_transport(x, mean_1, mean_2, covariance_1, covariance_2):
    """ Gaussian tranport.

    Parameters
    ----------
    x: data to transport.
        torch tensor shape=(n_samples, n_components)

    mean_1: mean of the source Gaussian.

    mean_2: mean of the target Gaussian.

    covariance_1: covariance of the source Gaussian.

    covariance_2: covariance of the target Gaussian.


    Returns
    -------

    x_transported: transported data.
        torch tensor shape=(n_samples, n_components)
    """
    M = plan_gaussian_transport(covariance_1, covariance_2)
    return mean_2 + (x - mean_1).mm(M)


def bures_Wasserstein(a_cov, b_cov):
    """ Minimum of the Frobenius norm of the covariance roots. 
    """
    U, S, V = torch.svd(a_cov)
    S_root = S.clamp(min=0).pow(0.5)
    root_a_cov = U.mm(torch.diag(S_root)).mm(V.t())
    # Interactions
    cov_prod = root_a_cov.mm(b_cov).mm(root_a_cov)
    return cov_prod



def bures_Wasserstein_distance(a_cov, b_cov, a_mean=None, b_mean=None):
    """ Computes the Bures-Wasserstein distance.

    Parameters
    ----------
    a_cov: covariance of the source Gaussian.
        torch tensor shape=(n_samples, n_components)

    b_cov: covariance of the target Gaussian.
        torch tensor shape=(n_samples, n_components)

    a_mean: mean of the source Gaussian.
        torch tensor shape=(n_components)

    b_mean: mean of the target Gaussian.
        torch tensor shape=(n_components)

    Returns
    -------
    dist: the Bures-Wasserstein distance.
    """

    mseloss = nn.MSELoss(reduction='sum')
    if (a_mean is not None) and (b_mean is not None): 
        mean_diff_squared = mseloss(a_mean, b_mean)
    else:
        mean_diff_squared = 0.0

    tr_a_cov = torch.trace(a_cov)
    tr_b_cov = torch.trace(b_cov)

    cov_prod = bures_Wasserstein(a_cov, b_cov)

    _, S, _ = torch.svd(cov_prod)
    var_overlap = S.clamp(min=0).pow(0.5).sum()

    dist = (mean_diff_squared + tr_a_cov + tr_b_cov - 2 * var_overlap)

    return dist


def gaussian_barycenter(means, covs, lambds, n_components=2,
                        tol=1e-5, max_iter = 100, verbose=True):
    """ Computes the Bures-Wasserstein barycenter.
    The algorithm starts with an initial guess of the Frechet mean; 
    it then lifts all observations to the tangent space at that
    initial guess via the log map, and averages linearly on the tangent space; this
    linear average is then retracted onto the manifold via the exponential map
    providing the next guess, and iterates.

    Parameters
    ----------

    means: means of the Gaussians to average. 
        torch tensor shape=(n_components, elements)

    covs: covatiances of the Gaussians to average.
        torch tensor shape=(n_components, n_components, elements)

    n_components: the dimensionality of the input data. 
        It corresponds to the number of features. int.

    tol: tolerance. float (optional)

    max_iter: maximum number of iterations. int (optional)

    verbose: flag to disply progress. bool (optional)


    Returns
    -------
    mean_k: mean of the barycenter.
        torch tensor shape=(n_components)

    cov_k: covariance matrix of the barycenter.
        torch tensor shape=(n_components, n_components).

        
    [1] V. Masarotto, V. M. Panaretos, and Y. Zemel. Procrustes metrics on covariance operators and optimal
        transportation of gaussian processes. Sankhya A, 2018.
    """
    
    ## Init
    cov_k = torch.eye(n_components)
    n_covs = len(covs)
    trace_ = n_components
    loop = 1
    cpt = 0

    ### Barycenter of means
    mean_k = lambds[0] * means[0] 
    for i in range(1, n_covs):
        mean_k = mean_k + lambds[i] * means[i] 

    ### Barycenter of covariances
    while loop:
        M = []
        for _, cov in zip(lambds, covs):
            M.append(plan_gaussian_transport(cov_k, cov, t=0.5))

        T_k = M[0] * lambds[0]
        for i in range(1, n_covs):
            T_k = T_k + M[i] * lambds[i]

        cov_k = T_k.matmul(cov_k.matmul(T_k))
        # Enforce symmetry
        cov_k = (cov_k + cov_k.t()) / 2.

        trace_cov_k = torch.trace(cov_k)

        ## Relatice error
        error = (trace_cov_k - trace_).abs() / trace_cov_k
        if error.data.numpy() < tol:
            loop = 0

        if cpt > max_iter:
            loop = 0

        trace_ = trace_cov_k
        cpt += 1

        if verbose and (cpt % 2 == 0):
            print("iteration: %s    relative error: %.4f" % (cpt, error.data.numpy()))
    
    return mean_k, cov_k