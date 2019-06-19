import numpy as np 
import torch
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import euclidean_distances
from gaussian_transport import (gaussian_barycenter, l2_transport)
import time
from sklearn.base import BaseEstimator
from gaussian_transport import plan_gaussian_transport
from sklearn.mixture import GaussianMixture


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    C = torch.sum(torch.abs(x_col - y_lin).pow(p), 2)
    return C


def get_assignment(gmm_x, gmm_y):
    """Compute linear assignment with the Hungarian algorithm.
    """
    M = euclidean_distances(
        gmm_x.means_, gmm_y.means_, squared=True)
    assignment = linear_assignment(M)
    return assignment


def get_pairwise_barycenter(gmm_x, gmm_y, assignment_xy, lambds):
    """Computes the pairwise barycenter. 
    The barycenter of Gaussians if the match condition is satisfied.
    
    Parameters
    ----------
    
    gmm_x: GaussianMixture of data X.
    
    gmm_y: GaussianMixture of data Y.
    
    assignment_xy: Matching condition.
    
    lambds: Weights of the barycenter.
    
    Returns
    --------
    
    means_k: list of means
    
    covs_k: list of covariances
    
    """
    ## number of features
    n_components = gmm_x.means_.shape[1]
    
    covs_k = []
    means_k = []
    ## Compute Bures-Wasserstein barycenter for matched pairs.
    for i, j in zip(*assignment_xy):
        
        mean_x = torch.tensor(gmm_x.means_[i]).unsqueeze(0).float()
        mean_y = torch.tensor(gmm_y.means_[j]).unsqueeze(0).float()
        
        cov_x = torch.tensor(gmm_x.covariances_[i]).unsqueeze(0).float()
        cov_y = torch.tensor(gmm_y.covariances_[j]).unsqueeze(0).float()
        
        means = torch.cat([mean_x, mean_y])
        covs = torch.cat([cov_x, cov_y])

        mean_k, cov_k = gaussian_barycenter(
            means, covs, lambds, n_components=n_components, verbose=False)
        covs_k.append(cov_k.data.numpy())
        means_k.append(mean_k.data.numpy())    
        
    return means_k, covs_k


def transport_samples_to_barycenter(
    X, Y, gmm_x, gmm_y, assignment_xy, means_k, covs_k):

    assign_x = gmm_x.predict(X)
    assign_y = gmm_y.predict(Y)

    X_transported = np.zeros_like(X)
    Y_transported = np.zeros_like(Y)    

    X_torch = torch.tensor(X).float()
    Y_torch = torch.tensor(Y).float()

    for k, (i, j) in enumerate(zip(*assignment_xy)):

        ind_x = np.where(assign_x == i)[0]
        ind_y = np.where(assign_y == j)[0]

        X_ = X_torch[ind_x, :]
        Y_ = Y_torch[ind_y, :]

        mean_x = torch.tensor(gmm_x.means_[i]).float()
        mean_y = torch.tensor(gmm_y.means_[j]).float()
        
        ## Covariances
        cov_x = torch.tensor(gmm_x.covariances_[i]).float()
        cov_y = torch.tensor(gmm_y.covariances_[j]).float()
        
        ## Transport
        X_transported[ind_x, :] = l2_transport(
            X_, mean_x, torch.tensor(means_k[k]).float(), 
            cov_x, torch.tensor(covs_k[k]).float()).data.numpy()
        
        Y_transported[ind_y, :] = l2_transport(
            Y_, mean_y, torch.tensor(means_k[k]).float(), 
            cov_y, torch.tensor(covs_k[k]).float()).data.numpy() 
        
    return X_transported, Y_transported


def get_emd_barycenter(X, Y, weights,
                       n_components=20, random_state=4):
    n_features = X.shape[1]
    
    rnd = check_random_state(seed=random_state)
    
    # n_components: number of Diracs of the barycenter
    measures_locations = []
    measures_weights = []
    for data in [X, Y]:
        n_i = data.shape[0]
        b_i = rnd.uniform(0., 1., (n_i,))
        b_i = b_i / np.sum(b_i)  # Dirac weights

        measures_locations.append(data)
        measures_weights.append(b_i)

    X_init = np.random.normal(0., 1., (n_components, n_features))  # initial Dirac locations
    b = np.ones((n_components,)) / float(n_components)  # weights of the barycenter (it will not be optimized, only the locations are optimized)

    X_bary = ot.lp.free_support_barycenter(
        measures_locations, measures_weights, 
        X_init, b, 
        weights=weights
    )
    
    return X_bary


class MappingTransportEstimator(BaseEstimator):
    """

    """
    def __init__(self, kernel="linear", 
                 bias=True, max_iter=20, sigma=1,
                 mu=1, eta=1e-8, n_components=20, 
                 log=False, path=None, barycenter=None):
        self.kernel = kernel
        self.bias = bias
        self.max_iter = max_iter
        self.mu = mu
        self.eta = eta
        self.log = log
        self.sigma = sigma
        self.n_components = n_components
        self.path = path
        self.barycenter = barycenter
        
        ## Tranport mapper (one per group)
        self.ot_map_a_ = ot.da.MappingTransport(
            kernel=kernel, mu=mu, eta=eta, sigma=sigma, bias=bias,
            max_iter=max_iter, verbose=False, log=log) 
        self.ot_map_b_ = ot.da.MappingTransport(
            kernel=kernel, mu=mu, eta=eta, sigma=sigma, bias=bias,
            max_iter=max_iter, verbose=False, log=log)         
    
    def fit(self, X, z):
        X = X.data.numpy()
        z = z.data.numpy()
        
        ind_a = np.where(z == 1)[0]
        ind_b = np.where(z != 1)[0]
            
        X_a = X[ind_a, :]
        X_b = X[ind_b, :]
                    
        if self.barycenter is not None:
            self.barycenter_ = self.barycenter
        else: 
            self.ti_barycenter_ = time.time()
            self.barycenter_ = get_emd_barycenter(
                X_a, X_b, weights=np.array([0.5, 0.5]), 
                n_components=self.n_components)
            self.to_barycenter_ = time.time()
        
            joblib.dump(self.barycenter_, self.path)
            joblib.dump(self.to_barycenter_ - self.ti_barycenter_, self.path + "*time")

        ## Fit map
        self.ti_transport_ = time.time()
        self.ot_map_a_.fit(Xs=X_a, Xt=self.barycenter_)
        self.ot_map_b_.fit(Xs=X_b, Xt=self.barycenter_)
        self.to_transport_ = time.time()        
        return self
    
    def transform(self, X, z):
        X = X.data.numpy()
        z = z.data.numpy()
        
        ind_a = np.where(z == 1)[0]
        ind_b = np.where(z != 1)[0]
            
        X_a = X[ind_a, :]
        X_b = X[ind_b, :]
        
        X_transported = np.zeros_like(X)
        trans_X_a = self.ot_map_a_.transform(Xs=X_a)
        trans_X_b = self.ot_map_b_.transform(Xs=X_b)
        X_transported[ind_a, :] = trans_X_a
        X_transported[ind_b, :] = trans_X_b
        return X_transported



class LBWMixEstimator(BaseEstimator):
    """Mixing sensitive groups estimator.
    
    Parameters
    ----------
    n_components: number of components for each Gaussian Mixture Model (GMM). 
        Int (optional).
        
    lambds: weights of the Wasserstein barycenter.
        They have to sum to 1.
        array-like, shape=(n_densities).
    
    random_state: seed for the GMM.
        Int (optional).
        
    Attributes
    -----------
    
    gmm_x_: sklearn GMM of group A.
    
    gmm_y_: sklearn GMM of group B.
    
    assignment_: Matching encoding of groups A and B.

    """
    def __init__(self, n_components=2, lambds=None,
                 random_state=None):
        self.n_components = n_components
        self.lambds = lambds
        self.random_state = random_state
        
    def _fit_gmm(self, X, z):
        self.n_features_ = X.shape[1]
        
        X_ = X.data.numpy()
        z_ = z.data.numpy()
        
        self.ti_model_ = time.time()
        self.gmm_x_ = GaussianMixture(
            n_components=self.n_components, 
            random_state=self.random_state).fit(X_[z_ == 1, :])
        self.gmm_y_ = GaussianMixture(
            n_components=self.n_components, 
            random_state=self.random_state).fit(X_[z_ != 1, :])
        self.to_model_ = time.time()

        self.means_x_ = torch.tensor(
            self.gmm_x_.means_).float()
        self.means_y_ = torch.tensor(
            self.gmm_y_.means_).float()
        self.covs_x_ = torch.tensor(
            self.gmm_x_.covariances_).float()
        self.covs_y_ = torch.tensor(
            self.gmm_y_.covariances_).float()
        
    def get_score(self, X, z):
        ind_a = np.where(z == 1)[0]
        ind_b = np.where(z != 1)[0]
        X_a = X[ind_a, :].clone()
        X_b = X[ind_b, :].clone()
        ll_x = self.gmm_x_.score(X_a.data.numpy())
        ll_y = self.gmm_y_.score(X_b.data.numpy())
        return ll_x, ll_y
    
    def fit(self, X, z):
        """Fit parametes.
        
        X: Input data.
            torch tensor, shape=(n_samples, n_features).
            
        z: Sample membership indicator.
            Note that in this implementation z only has two possible values, [0, 1].
            torch tensor, shape=(n_samples).
        """
        ## Fit GMM
        self._fit_gmm(X, z)
        self.ti_params_ = time.time()
        self.M_ = cost_matrix(self.means_x_, self.means_y_).float()
        self.M_ /= self.M_.max()
        ## Assigment 
        self.assignment_ = linear_assignment(self.M_.data.numpy())
        self.to_params_ = time.time()

        ## Get barycenters
        self.ti_barycenter_ = time.time()        
        self._transport_samples(X, z, compute_barycenter=True)
        self.to_barycenter_ = time.time()        
        return self

    def _transport_samples(self, X, z, compute_barycenter=True):
        ind_a = np.where(z == 1)[0]
        ind_b = np.where(z != 1)[0]
            
        X_a = X[ind_a, :].clone()
        X_b = X[ind_b, :].clone()
        
        assign_x = self.gmm_x_.predict(X_a.data.numpy())
        assign_y = self.gmm_y_.predict(X_b.data.numpy())
        
        lambds = (self.lambds
                  if self.lambds is not None 
                  else torch.tensor([0.5, 0.5]).float()) 

        if compute_barycenter:
            self.means_k_ = []
            self.covs_k_ = []

        X_a_transported = torch.zeros_like(X_a)
        X_b_transported = torch.zeros_like(X_b)     
        
        for k, (i, j) in enumerate(
            zip(*self.assignment_)):
                                    
            ind_x = np.where(assign_x == i)[0]
            ind_y = np.where(assign_y == j)[0]
            X_a_ = X_a[ind_x, :]
            X_b_ = X_b[ind_y, :]
            mean_a = self.means_x_[i]
            mean_b = self.means_y_[j]
            ## Covariances
            cov_a = self.covs_x_[i]
            cov_b = self.covs_y_[j]
            
            if compute_barycenter:
                means = torch.cat([mean_a.unsqueeze(0), 
                                   mean_b.unsqueeze(0)])
                covs = torch.cat([cov_a.unsqueeze(0), 
                                  cov_b.unsqueeze(0)])
                mean_k, cov_k = gaussian_barycenter(
                    means, covs, lambds, 
                    n_components=self.n_features_, verbose=False)
                
                self.covs_k_.append(cov_k.float())
                self.means_k_.append(mean_k.float())  
                
            ## Transport
            X_a_transported[ind_x, :] = l2_transport(
                X_a_, mean_a, self.means_k_[k], cov_a, self.covs_k_[k])
            X_b_transported[ind_y, :] = l2_transport(
                X_b_, mean_b, self.means_k_[k], cov_b, self.covs_k_[k])
        
        X_transported = torch.zeros_like(X)
        X_transported[ind_a, :] = X_a_transported
        X_transported[ind_b, :] = X_b_transported
        return X_transported.data.numpy()
    
    def transform(self, X, z):
        return self._transport_samples(X, z, compute_barycenter=False)