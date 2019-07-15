import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

import gpopt

from copy import copy

class LoNStaKernel(gpopt.Kernel):
    def __init__(self, kernel_global, kernel_local, pos_local, h_local, a_local=1):
        self.kernel_global = kernel_global
        self.kernel_local = kernel_local
        
        self.pos_local = pos_local
        self.h_local = h_local
        self.a_local = a_local
        
        params_global = self.kernel_global.get_params()
        params_local = self.kernel_local.get_params()
        
        self.kernel_global_nparams = len(params_global)
        self.kernel_local_nparams = len(params_local)
    
    def get_nu(self, pts):
        diffs = np.asarray(pts) - self.pos_local
        return self.a_local * np.sqrt(2 * np.pi) * sp.stats.norm.pdf(np.linalg.norm(diffs / self.h_local, axis=-1))
    
    def evaluate(self, ptsA, ptsB=None):
        eval_global = self.kernel_global.evaluate(ptsA, ptsB)
        eval_local = self.kernel_local.evaluate(ptsA, ptsB)
        
        if ptsB is None:
            ptsB = ptsA    
        
        nuA = self.get_nu(ptsA)
        nuB = self.get_nu(ptsB)
        
        wlocA = nuA / (1 + nuA)
        wlocB = nuB / (1 + nuB)
        
        eval_local = np.sqrt(wlocA)[:,np.newaxis] * eval_local * np.sqrt(wlocB)[np.newaxis,:]
        eval_global = np.sqrt(1 - wlocA[:,np.newaxis]) * eval_global * np.sqrt(1 - wlocB[np.newaxis,:])
        
        ret = eval_local + eval_global
        return ret
    
    def get_params(self):
        params_global = self.kernel_global.get_params()
        params_local = self.kernel_local.get_params()
        
        self.kernel_global_nparams = len(params_global)
        self.kernel_local_nparams = len(params_local)
        
        params = np.concatenate([ params_global, params_local, 
                                  self.pos_local, np.log(self.h_local), [np.log(self.a_local)] ], axis=0)
        return params
    
    def split_params(self, args):
        assert self.kernel_global_nparams is not None
        split_idcs = [self.kernel_global_nparams, -len(self.pos_local) - len(self.h_local) - 1, - len(self.h_local) - 1, -1]
        return np.array_split(args, split_idcs)
        
    def create_from_params(self, args):
        p_glob, p_loc, new_pos_loc, new_logh_loc, new_loga_loc = self.split_params(args)
        
        new_kernel_global = self.kernel_global.create_from_params(p_glob)
        new_kernel_local = self.kernel_local.create_from_params(p_loc)
        
        new_loga_loc = new_loga_loc[0]
        
        return LoNStaKernel(new_kernel_global, new_kernel_local, new_pos_loc, np.exp(new_logh_loc), np.exp(new_loga_loc))
    
    def __str__(self):
        return "LoNStaKernel(kernel_global={}, kernel_local={}, pos_local={}, h_local={}, a_local={})".format(
                str(self.kernel_global), str(self.kernel_local), self.pos_local, self.h_local, self.a_local)
    
    def get_bounds(self, bounds_global=None, bounds_local=None, pos_bounds=None):
        assert self.kernel_global_nparams is not None
        
        if bounds_global is None:
            bounds_global = [ (None, None) for i in range(self.kernel_global_nparams) ]
            
        if bounds_local is None:
            bounds_local = [ (None, None) for i in range(self.kernel_local_nparams) ]
            
        if pos_bounds is None:
            pos_bounds = [ (None, None) for p in self.pos ]
            
        rem_bounds = [ (None, None) for x in self.h_local ] + [ (None, None) ]
        ret = [ x for x in bounds_global ] \
              + [ x for x in bounds_local ] \
              + [ x for x in pos_bounds ] + rem_bounds
                
        return ret
            
    def get_log_prior(self, prior_global=None, prior_local=None, pos_means=None, pos_scales=1, h_means = 1, h_scales = 1, a_scale=1):
        if prior_global is None:
            prior_global = lambda x: 0
            
        if prior_local is None:
            prior_local = lambda x: 0
            
        def log_prior(args):    
            p_glob, p_loc, new_pos_loc, new_logh_loc, new_loga_loc = self.split_params(args)
            
            ret = prior_global(p_glob) + prior_local(p_loc)
            
            if pos_means is not None:
                ret -= 0.5 * np.sum(((new_pos_loc - pos_means) / pos_scales) ** 2)
                
            ret -= 0.5 * np.sum(((new_logh_loc - np.log(h_means)) / h_scales) ** 2)
            ret -= 0.5 * ((new_loga_loc[0] / a_scale) ** 2)
            
            return ret
                
        return log_prior
    
