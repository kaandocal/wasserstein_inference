import numpy as np
import scipy as sp
import scipy.misc
import scipy.optimize

import time
import dill

import utils

import logging
logger = logging.getLogger(__name__)

def logmatvec(mat, vec):
    """ Matrix-vector multiplication in log-space """
    sum_shape = [ *mat.shape ] + [ 1 for i in range(vec.ndim - 1) ]
    sums = mat.reshape(sum_shape) + vec[np.newaxis,...]
    
    ret = sp.misc.logsumexp(sums, axis=1)
    return ret

def logmatvec_sepkernel(v, log_Ks):
    """ Matrix-vector multiplication in log-space for separable matrices """
    for i in range(v.ndim):
        v = np.rollaxis(v, -1)
        v = logmatvec(log_Ks[-i-1], v)

    return v

### Code for overrelaxed Sinkhorn algorithm
def phi(log_x, omega):
    return np.exp(log_x) * (1 - np.exp(-omega * log_x)) - omega * log_x

def Theta_star(log_u):
    omegas = np.linspace(1, 2, 51)
    phis = phi(log_u, omegas)
    idcs = np.where(phis >= 0)[0]
    
    if len(idcs) == 0:
        ret = 1
    else:
        ret = omegas[idcs[-1]]
        
    return ret
    
def Theta(log_u, delta=0.01, theta0=1.5):
    min_log_u = np.min(log_u)
    return min(theta0, max(1, Theta_star(min_log_u) - delta))    

###

class ParticleDistribution:
    """ Particle distribution (histogram) class
        
        Attributes:
            n_species: int
                Number of species (histogram dimension)
                    
            bounds: array of ints
                Dimensions of the histogram
                    
            mean: array of floats
                Mean of the histogram
    """
    def __init__(self, array, weights=None, hist=False):
        if not hist:
            self.data = self.data_from_array(array, weights=weights)
        else:
            self.data = self.data_from_hist(array)
            
        self.bounds = np.array(self.data.shape) - 1
        self.n_species = self.data.ndim
        
        slices = [ slice(0, bound+1, 1) for bound in self.bounds ]
        
        self.idcs = np.rollaxis(np.mgrid[slices], 0, self.n_species+1)
        
        E = np.eye(self.n_species)
        self.mean = np.asarray([ self.noncentral_moment(e) for e in E ])
        
    def data_from_array(self, array, weights=None):
        array = np.asarray(array, dtype=int)
        
        assert array.ndim == 2 or array.ndim == 1
        assert np.all(array >= 0)
        
        if array.ndim == 1:
            array = array[np.newaxis,:]
            
        bounds = np.max(array, axis=-1)
        data = np.zeros(bounds + 1)
        
        if weights is not None:
            weights = np.asarray(weights)
            
            assert np.all(weights >= 0)
            assert weights.shape == (array.shape[1],)
        
            for ss, wt in zip(array.T, weights):
                data[tuple(ss)] += wt
        else:
            for ss in array.T:
                data[tuple(ss)] += 1
            
        data /= np.sum(data)
        return data
    
    def data_from_hist(self, hist):
        return np.asarray(hist) / np.sum(hist)
    
    def coarsen(self, dim, scale=2):        
        """ Reduce histogram dimensions by binning counts in the specified dimension
            by the given scale factor """
        data_reord = np.moveaxis(self.data, dim, 0)
        
        new_shape = [ si for si in data_reord.shape ]
        new_shape[0] = int(np.ceil(new_shape[0] / scale))
        
        arrays = np.empty((scale, *new_shape))
        
        for i in range(scale):
            nsize = 1 + (self.bounds[dim] - i) // scale
            
            arrays[i,-1] = 0
            arrays[i,:nsize] = data_reord[i::scale]
        
        arrays_reord = np.moveaxis(arrays, 1, dim+1)
        hist_new = np.sum(arrays_reord, 0)
        return ParticleDistribution(hist_new, hist=True)
        
    def marginal(self, spec):
        """ Construct a lower-dimensional histogram by marginalizing over
            all dimensions not specified in the argument """
        if np.isscalar(spec):
            spec = [spec]s
            
        sum_axes = tuple([ i for i in range(self.n_species) if i not in spec ])
        hist_new = np.sum(self.data, axis=sum_axes)
        
        return ParticleDistribution(hist_new, hist=True)
        
    def noncentral_moment(self, exponents):
        assert len(exponents) == self.n_species
                
        ret = np.sum(self.data * np.prod(self.idcs ** exponents, axis=-1))
        return ret
    
    def central_moment(self, exponents):
        assert len(exponents) == self.n_species
                
        ret = np.sum(self.data * np.prod((self.idcs - self.mean) ** exponents, axis=-1))
        return ret
    
    def wasserstein_dist_1D(self, other, p=1, weights=None):
        assert self.n_species == 1
        assert self.n_species == other.n_species
        
        if weights is None:
            weights = 1
            
        weights = np.asscalar(np.asarray(weights))
            
        mass_other = np.copy(other.data)
        mass_other_idxmin = 0
        
        cost = 0
        
        for i in range(self.bounds[0] + 1):
            moved_to_i = 0
            
            for j in range(mass_other_idxmin, len(mass_other)):
                if mass_other[j] + moved_to_i <= self.data[i]:
                    moved_to_i += mass_other[j]
                    cost += mass_other[j] * (np.abs(j - i) ** p) * weights
                    mass_other[j] = 0
                    mass_other_idxmin = j + 1
                    continue
                else:
                    amt = self.data[i] - moved_to_i
                    mass_other[j] -= amt
                    cost += amt * (np.abs(j - i) ** p) * weights
                    break
                    
        return cost ** (1 / p)
    
    def wasserstein_dist(self, other, p=1, weights=None, **kwargs):
        if self.n_species == 1:
            return self.wasserstein_dist_1D(other, p=p, weights=weights, **kwargs)
        else:
            return self.wasserstein_dist_sinkhorn_wrapper(other, p=p, weights=weights, **kwargs)
    
    def wasserstein_dist_sinkhorn_wrapper(self, other, weights=None, p=1, **kwargs):s
        if self.n_species == 1:
            return self.wasserstein_dist_1D(other, p=p, weights=weights, **kwargs)
        
        if weights is None:
            weights = np.ones(self.n_species)
            
        try:
            max_dim = np.max(np.maximum(self.bounds, other.bounds))
            if max_dim > 1000:
                raise MemoryError            # Precaution
                
            ret = self.wasserstein_dist_sinkhorn(other, p=p, weights=weights, **kwargs)
        except MemoryError:
            logging.info("--- OUT OF MEMORY ---")
            
            max_dim = np.argmax(np.maximum(self.bounds, other.bounds))
            
            logging.info("Will coarsen histograms in dimension {}".format(max_dim))
            
            self_subs = self.coarsen(max_dim, 2)
            other_subs = other.coarsen(max_dim, 2)
            
            weights_subs = np.copy(weights)
            weights_subs[max_dim] *= 2
            
            ret = self_subs.wasserstein_dist_sinkhorn_wrapper(other_subs, p=p, weights=weights_subs, **kwargs)
            
            logging.info("--- END OUT OF MEMORY ---")
            
        return ret

    def wasserstein_dist_sinkhorn(self, other, p=1, eps_max=10, eps_min=0.03, n_iter=10, weights=None, **or_kwargs):
        Cs = [ np.abs(np.arange(b_self + 1)[:,np.newaxis] - np.arange(b_other + 1)[np.newaxis,:]) ** p 
               for b_self, b_other in zip(self.bounds, other.bounds) ]

        if weights is None:
            weights = np.ones(len(Cs))

        Cs = [ C * (w ** p) for C, w in zip(Cs, weights) ]

        delta = (eps_min / eps_max) ** (1 / n_iter)
        eps_list = [ eps_max * (delta ** i) for i in range(n_iter + 1) ]

        log_f = -np.log(self.data.size) * np.ones_like(self.data)
        log_g = -np.log(other.data.size) * np.ones_like(other.data)

        log_self_data = np.log(self.data + 1e-15)
        log_other_data = np.log(other.data + 1e-15)

        costs = []

        for eps in eps_list:
            log_Ks = [ -C / eps for C in Cs ]
            log_KTs = [ log_K.T for log_K in log_Ks ]

            while True:
                log_f_tilde = -eps * logmatvec_sepkernel(log_g / eps + log_other_data, log_Ks)
                omega_f = Theta((log_f - log_f_tilde) / eps, **or_kwargs)
                log_f = (1 - omega_f) * log_f + omega_f * log_f_tilde

                log_g_tilde = -eps * logmatvec_sepkernel(log_f / eps + log_self_data, log_KTs)
                omega_g = Theta((log_g - log_g_tilde) / eps, **or_kwargs)
                log_g = (1 - omega_g) * log_g + omega_g * log_g_tilde

                log_a = log_f / eps + log_self_data
                log_b = log_g / eps + log_other_data

                dst = np.exp(log_b + logmatvec_sepkernel(log_a, log_KTs))
                diff = np.sum(np.abs(dst - other.data))
                if diff < 0.01:
                    break

            cost = 0

            for i in range(log_a.ndim):
                F = log_b

                for j in range(log_a.ndim):
                    F = np.rollaxis(F, -1)

                    K = log_Ks[-j-1]

                    if j == i:
                        K = K + np.log(Cs[-j-1])

                    F = logmatvec(K, F)

                cost += np.sum(np.exp(log_a + F))

            cost = cost ** (1 / p)
            costs.append(cost)

        cost = costs[-1]
        return cost
