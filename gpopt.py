import numpy as np
import scipy as sp
import scipy.special, scipy.optimize, scipy.stats

import dill

import logging
logger = logging.getLogger(__name__)

class Kernel:        
    """ GP kernel base class """
    def evaluate(self, ptsA, ptsB=None):
        """ Evaluate kernel at the given points
        
            Arguments:
                ptsA: array of n vectors
                ptsB: array of m vectors (default: ptsA)
                
            Returns:
                n x m kernel matrix
            """
        raise NotImplementedError()
        
    def evaluate_diag(self, ptsA):
        return np.diag(self.evaluate(ptsA))
    
class NonisotropicGaussianKernel(Kernel):
    def __init__(self, sigma_y, h):
        self.sigma_y = sigma_y
        self.h = h
        
    def evaluate(self, ptsA, ptsB=None):
        if ptsB is None:
            ptsB = ptsA
            
        ptsA = np.asarray(ptsA)
        ptsB = np.asarray(ptsB)
            
        diffs = ptsA[:,np.newaxis] - ptsB[np.newaxis,:]
        dists2 = np.sum((diffs / self.h) ** 2, axis=-1)
        
        return (self.sigma_y ** 2) * np.exp(-0.5 * dists2)
    
    def __str__(self):
        return "NonisotropicGaussianKernel(sigma_y={}, h={})".format(self.sigma_y, self.h)
    
    def get_params(self):
        return [ np.log(self.sigma_y), *np.log(self.h) ]
    
    def create_from_params(self, args):
        ls, *lh = args
        return NonisotropicGaussianKernel(sigma_y=np.exp(ls), h=np.exp(lh))
    
    @staticmethod
    def get_log_prior(sigma_mean = 1, sigma_scale = 1, h_mean = 1, h_scale = 1):
        def log_prior(args):
            ls, *lh = args

            ret = -0.5 * ((ls - np.log(sigma_mean)) / sigma_scale) ** 2 \
                  -0.5 * np.sum(((lh - np.log(h_mean)) / h_scale) ** 2)

            return ret

        return log_prior

class GP:
    def __init__(self, d, kernel, obs_noise, mean=0):
        self.d = d
        self.kernel = kernel
        
        self.obs_x = np.empty((0,d))
        self.obs_y = np.empty((0))
        self.obs_data = []
        
        self.Sigma_obs = np.empty((0,0))
        self.iSigma_obs = np.empty((0,0))
        
        self.obs_noise = obs_noise
        
        self.mean = mean
        
        self.kernels_hist = []
        
    def _evaluate(self, points, obs_x, obs_y, iSigma_obs, mean_only=False, with_obs_noise=True):
        Sigma_is = self.kernel.evaluate(points, obs_x)
        Sigma_pts = np.asarray([ self.kernel.evaluate([pt]) for pt in points ]).reshape(len(points))
        
        mean_corr = np.dot(iSigma_obs, obs_y - self.mean)
        means = self.mean + np.dot(Sigma_is, mean_corr)
        
        if mean_only:
            return means
        
        if len(obs_x) == 0:
            cov_corr = 0
        else:
            cov_corr = np.einsum("ij,jk,ki->i", Sigma_is, iSigma_obs, Sigma_is.T)
            
        covs = Sigma_pts - cov_corr
        if with_obs_noise:
            covs = np.sqrt(covs + self.obs_noise ** 2)
        else:
            covs = np.sqrt(covs)
            
        return means, covs
    
    def negloglikelihood(self, kernel=None, obs_noise=None):
        if kernel is None and obs_noise is None:
            Sigma_obs = self.Sigma_obs
            iSigma_obs = self.iSigma_obs
        else:
            Sigma_obs = self.compute_Sigma(self.obs_x, kernel=kernel, obs_noise=obs_noise)
            iSigma_obs = np.linalg.inv(Sigma_obs) 
        
        diffs = self.obs_y - self.mean
        T = np.dot(iSigma_obs, diffs)
        expterm = np.dot(diffs, T)
        
        _, logdet = np.linalg.slogdet(Sigma_obs)
        
        return 0.5 * (logdet + expterm + len(self.obs_y) * np.log(2 * np.pi))
        
    def history(self, n, with_kernel=True):
        obs_x_new = self.obs_x[:n]
        obs_y_new = self.obs_y[:n]
        obs_data_new = self.obs_data[:n]
        
        if with_kernel and hasattr(self, "kernels_hist") and len(self.kernels_hist) > 0:
            for r, k in self.kernels_hist:
                kernel = k
                if r >= n:
                    break
                    
            if r < n:
                kernel = self.kernel
        else:
            kernel = self.kernel
            
        ret = GP(d=self.d, kernel=kernel, obs_noise=self.obs_noise, mean=self.mean)
        
        if n != 0:
            ret.add_obs(obs_x_new, obs_y_new, obs_data_new)
        
        return ret
        
    def evaluate(self, points, **kwargs):
        points = np.asarray(points)
        assert points.shape[-1] == self.d
        
        return self._evaluate(points, self.obs_x, self.obs_y, self.iSigma_obs, **kwargs)
    
    def evaluate_with(self, points, obs_x, obs_y, **kwargs):
        points = np.asarray(points)
        assert points.shape[-1] == self.d
        
        obs_x = np.concatenate((self.obs_x, obs_x), axis=0)
        obs_y = np.concatenate((self.obs_y, obs_y), axis=0)
        
        Sigma_obs = self.compute_Sigma(obs_x)
        iSigma_obs = np.linalg.inv(Sigma_obs)
        
        return self._evaluate(points, obs_x, obs_y, iSigma_obs, **kwargs)
        
    def compute_Sigma(self, xx, kernel=None, obs_noise=None):
        if kernel is None:
            kernel = self.kernel
            
        if obs_noise is None:
            obs_noise = self.obs_noise
            
        return kernel.evaluate(xx) + (obs_noise ** 2) * np.eye(len(xx))
        
    def add_obs(self, xx, yy, data=None):
        self.obs_x = np.concatenate((self.obs_x, xx), axis=0)
        self.obs_y = np.concatenate((self.obs_y, yy), axis=0)
        
        if data is None:
            data = [ None for x in xx ]
            
        self.obs_data = [ *self.obs_data, *data ]
        self.update_Sigma_obs()
        
        if np.linalg.cond(self.Sigma_obs) > 1e9:
            logger.warning("Ill-conditioned covariance matrix")
        
    def update_Sigma_obs(self):
        self.Sigma_obs = self.compute_Sigma(self.obs_x)
        self.iSigma_obs = np.linalg.inv(self.Sigma_obs)    
        
    def infer_mean(self):
        logger.info("--- Optimising GP mean ---")
        logger.debug("Current GP: \n{}\n".format(str(self)))
        self.mean = np.mean(self.obs_y)
        
    def fit_kernel(self, fit_mean=True, logprior=None, print_output=False, opt_kwargs={}):
        if len(self.obs_y) == 0:
            logger.warning("Cannot fit kernel to empty GP")
            return
        
        if fit_mean:
            self.infer_mean()
            
        kernel_new = self.train_kernel_hyperparameters(logprior=logprior, print_output=print_output, opt_kwargs=opt_kwargs)
        
        if hasattr(self, "kernels_hist"):
            self.kernels_hist.append([len(self.obs_y), self.kernel])
            
        self.kernel = kernel_new
            
        self.update_Sigma_obs()
        
    def train_kernel_hyperparameters(self, logprior=None, print_output=False, opt_kwargs={}):
        x0 = np.array(self.kernel.get_params())
        
        if logprior is None:
            def fun(args):
                kernel_new = self.kernel.create_from_params(args)
                return self.negloglikelihood(kernel=kernel_new)
        else:
            def fun(args):
                kernel_new = self.kernel.create_from_params(args)
                return self.negloglikelihood(kernel=kernel_new) - logprior(args)
        
        logger.info("--- Optimising kernel hyperparameters ---")
        logger.info("Current GP: \n{}\n".format(str(self)))
        if opt_kwargs:
            logger.info("opt_kwargs = {}".format(str(opt_kwargs)))
            
        res = sp.optimize.minimize(fun, x0, **opt_kwargs)
        logger.info("sp.optimize.minimize message: {}".format(res["message"]))
        logger.debug("sp.optimize.minimize returned:\n{}\n".format(str(res)))
        
        if print_output:
            print(res)
            
        args = res["x"]
        
        return self.kernel.create_from_params(args)
        
    def __str__(self):
        return "GaussianProcess(d={}, kernel={}, obs_noise={}, mean={})".format(self.d, str(self.kernel), self.obs_noise, self.mean)
        
class AcquisitionFunction:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        
    def evaluate(self, gp, points, grad=False):
        """ Will try to find minimum! """
        raise NotImplementedError()
        
    def optimize(self, gp, x0=None, **opt_kwargs):
        if x0 is None:
            if len(gp.obs_x) == 0:
                x0 = np.zeros(gp.d)
            else:
                idx = np.argmin(gp.obs_y)
                x0 = gp.obs_x[idx] + 0.00001 * self.rng.normal(size=gp.d)

        fun = self.evaluate
        
        old_state = np.random.get_state()
        np.random.seed(self.rng.randint(0, 2 ** 32 - 1))
        
        logger.info("--- Optimising AF ---")
        if opt_kwargs:
            logger.info("opt_kwargs = {}".format(str(opt_kwargs)))
            
        res = sp.optimize.basinhopping(fun, x0, minimizer_kwargs = {"args" : (gp,), **opt_kwargs})
        logger.info("sp.optimize.basinhopping status: {}".format(res["message"]))
        logger.debug("sp.optimize.basinhopping returned:\n{}\n".format(str(res)))
        np.random.set_state(old_state)
        return res["x"], res["fun"]
    
class LogExpectedImprovement(AcquisitionFunction):
    def __init__(self, jitter=0.01, seed=None):
        super().__init__(seed=seed)
        
        self.jitter = jitter
    
    def evaluate(self, points, gp):        
        """ This long-winded function just computes the log EI in
            a numerically stable way """
        points = np.asarray(points)
        
        if len(gp.obs_x) == 0:
            return np.zeros(points.shape[:-1])
        
        obs_min = np.min(gp.obs_y)
        
        points_f = points.reshape((-1, gp.d))
        means, SS = gp.evaluate(points_f)
        
        SS[SS < 1e-6] = 1e-6
        
        Z = (obs_min - means - self.jitter) / SS
        
        logPhi = sp.stats.norm.logcdf(Z)
        logpdf = sp.stats.norm.logpdf(Z)
        
        idcs_n, = np.where(Z < 0)
        idcs_z, = np.where(Z == 0)
        idcs_p, = np.where(Z > 0)
        
        t1_p = np.log(obs_min - means[idcs_p] - self.jitter) + logPhi[idcs_p]
        t1_n = np.log(-(obs_min - means[idcs_n] - self.jitter)) + logPhi[idcs_n]
        t2 = np.log(SS) + logpdf
        
        ret = np.zeros(means.shape[0])
        
        ret[idcs_z] = t2[idcs_z]
        ret[idcs_n] = t2[idcs_n] + np.log(1 - np.exp(t1_n - t2[idcs_n]))
        
        idcs_pa, = np.where(t1_p >= t2[idcs_p])
        idcs_pb, = np.where(t1_p < t2[idcs_p])
        
        ret[idcs_p[idcs_pa]] = t1_p[idcs_pa] + np.log(1 + np.exp(t2[idcs_p][idcs_pa] - t1_p[idcs_pa]))
        ret[idcs_p[idcs_pb]] = t2[idcs_p][idcs_pb] + np.log(1 + np.exp(t1_p[idcs_pb] - t2[idcs_p][idcs_pb]))
        
        ret = ret.reshape(points.shape[:-1])
            
        return -np.sign(ret) * np.log(1 + np.abs(ret))
    
    def __str__(self):
        return "ExpectedImprovement(jitter={}, seed={})".format(self.jitter, self.seed)
    
class Model:
    """ Model base class """
    def run_single(*args, **kwargs):
        return NotImplementedError()

class Trainer:
    def __init__(self, gp, model, af, ranges, init_obs = 20, n_iter = 5, model_kwargs = {}, out=None, seed=None):
        """ Selected arguments:
                ranges: d x 2 array of floats
                    Search range in the form [min,max] for every dimension
                    
                init_obs: int
                    Number of initial points for pre-training the GP
                    
                n_iter: int
                    Number of restarts during AF optimization (to prevent local optima)
                    
                out: string
                    Filename to save results in
        """
        self.gp = gp
        
        self.model = model
        self.model_kwargs = model_kwargs
        
        self.af = af
        self.ranges = np.asarray(ranges)
        
        assert len(self.ranges) == self.gp.d
        
        self.x_opt = None
        
        self.rng = np.random.RandomState(seed=seed)
        
        self.init_obs = init_obs
        self.n_iter = n_iter
        
        self.out = out
        
    def draw_random_points(self, n):
        """ Sample n points in search region using Latin hypercubes """
        idcs = np.asarray([ self.rng.permutation(n) for i in range(self.gp.d) ])
        idcs = idcs.T
        
        box_size = (self.ranges[:,1] - self.ranges[:,0]) / n
        
        assert len(box_size) == self.gp.d
        
        uu = self.rng.uniform(0, 1, size=(n, self.gp.d))
        
        pts = self.ranges.T[0] + (uu + idcs) * box_size
        return pts
        
    def pretrain_gp(self):
        """ Pre-train GP by evaluating model at a specified number of 
            points covering the search region """
        if self.init_obs == 0:
            return
        
        pretrain_pts = self.draw_random_points(self.init_obs)
        self.train_at(pretrain_pts)
        
    def comp_next_point(self, n_iter=None):
        """ Compute next point to evaluate the model by optimizing the
            acquisition function 
            
            Arguments:
                n_iter: int
                    Number of restarts for the optimization procedure
            """
        if n_iter is None:
            n_iter = self.n_iter
            
        # Start in the center of the search region if there are 
        # no observations at all
        if len(self.gp.obs_x) == 0:
            self.x_opt = np.mean(self.ranges, axis=-1)
                
            return self.x_opt
        
        x_min = None
        y_min = float("inf")
        
        x0s = self.draw_random_points(n_iter)
        for x0 in x0s:
            x, y = self.af.optimize(self.gp, x0, bounds=self.ranges)
            
            if y < y_min:
                x_min = x
                y_min = y
            
        self.x_opt = x_min
        return self.x_opt
    
    def train(self, n_rounds):
        """ Run Bayesian optimization for given number of rounds """
        for i in range(n_rounds):
            x_opt = self.comp_next_point()
            self.train_at([x_opt])
                
    def train_at(self, points):
        """ Utility function: train GP at selected points by running the simulator """
        for point in points:
            y, data = self.model.run_single(point, **self.model_kwargs)
            
            self.gp.add_obs([point], [np.asscalar(y)], data=[data])

            if self.out is not None:
                with open(self.out, "wb") as of:
                    dill.dump(self, of)