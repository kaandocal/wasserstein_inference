import numpy as np
import cme
import pdist

import time
from copy import copy

import logging
logger = logging.getLogger(__name__)

class SummaryStatistic:
    def compute(self):
        raise NotImplementedError()

class WassersteinDistance(SummaryStatistic):
    def __init__(self, marginals=None, weights=None, 
                 solver=pdist.ParticleDistribution.wasserstein_dist, solver_kwargs = {},
                 conv_marg=False):
        self.solver = solver
        self.marginals = marginals
        self.weights = weights
        self.solver_kwargs = {}
        self.conv_marg = conv_marg
        
    def compute(self, dist, ref_dist=None, dist_old=None):
        marginals = self.marginals
        ss = 0
        diff_ss = None
        
        if marginals is None:
            marginals = np.arange(dist.n_species)
                
        marg_dist = dist.marginal(marginals)
        
        if ref_dist is not None:
            marg_dist_ref = ref_dist.marginal(marginals)
            
            ss = self.solver(marg_dist, marg_dist_ref, weights=self.weights, **self.solver_kwargs)
        
        if dist_old is not None:
            marg_dist_old = dist_old.marginal(marginals)
            
            if self.conv_marg:
                diff_ss = 0
                for i in range(marg_dist_old.n_species):
                    marg_dist_old_i = marg_dist_old.marginal([i])
                    marg_dist_i = marg_dist.marginal([i])
                    diff_ss += self.solver(marg_dist_i, marg_dist_old_i, weights=[self.weights[i]], **self.solver_kwargs)
            else:
                diff_ss = self.solver(marg_dist, marg_dist_old, weights=self.weights, **self.solver_kwargs)
                    
        return ss, diff_ss
    
class SimModel:
    def __init__(self, n_species, reactions, summ_stats, initial_state=None, logtrans=True,
                 obs=None, ref_dist=None, logfile_prefix="log", sim_kwargs={}, seed=None):
        self.n_species = n_species
        self.reactions = reactions
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.compute_param_idcs()
        
        self.summ_stats = summ_stats
        
        self.initial_state = initial_state
        
        self.obs = obs
        self.ref_dist = ref_dist
        
        self.create_logfile(logfile_prefix)
        self.sim_kwargs = { "t_block" : 500 }
        
        self.logtrans = logtrans
        self.sim_kwargs.update(sim_kwargs)
        
    def run_single(self, params, **sim_kws):
        dist = self.evaluate(params, **sim_kws)
        obs, _ = self.compute_summ_stats(dist)
        ret = np.linalg.norm(obs - self.obs)
        self.log("Observed: {}".format(obs))
        
        if self.logtrans:
            ret = np.log(1 + ret)
            
        return ret, dist
        
    def evaluate(self, params, **sim_kws):
        kwargs = { k : v for k, v in self.sim_kwargs.items() }
        kwargs.update(sim_kws)
        
        params = np.asarray(params).reshape(-1)
        assert params.shape[0] == self.d_params
        
        part_seed = self.rng.randint(0, 2 ** 32 - 1)
        self.log("Running system with parameters: {}".format(params))
        self.log("Simulator kwargs: {}".format(sim_kws))
        self.log("Random seed: {}".format(part_seed))
        system = self.create_system(params)
        part = system.create_particle_system(seed=part_seed)

        dist = self.simulate(part, **kwargs)
        return dist
    
    def simulate(self, part, t_block=500, max_iter=20, tol=0.01, conv_iter=5, rel_es=False, **kwargs):
        ss = None
        conv_counter = 0
        
        n_part = None
        
        self.log("Simulating with t_block = {}\t\ttol = {}\t\t{}".format(t_block, tol, kwargs))
        self.log("Reaction rates: {}".format([r.rate for r in part.system.reactions]))
        
        dist_old = None
        for i in range(max_iter):
            dist = self.run_part(part, tmax=t_block, **kwargs)
            
            if i == 0:
                dist_old = dist
                continue
                
            ss, disc = self.compute_summ_stats(dist, dist_old)
            
            self.log("Current summary statistics: {}".format(ss))
            self.log("Current discrepancy: {}".format(disc))
            
            dist_old = dist
            
            if rel_es and self.obs is not None:
                conv_cond = (np.abs(disc) < tol * (ss - self.obs)) | (np.abs(disc) < tol)
            else:
                conv_cond = np.abs(disc) < tol
                
            if np.all(conv_cond):
                conv_counter += 1
                if conv_counter == conv_iter:
                    break
            else:
                conv_counter = 0

        if i == max_iter - 1:
            logger.warning("max_iter reached in SimModel.simulate")
        return dist
    
    def compute_summ_stats(self, dist, dist_old=None):
        ret_ss = np.empty(len(self.summ_stats))
        ret_disc = None
        
        if dist_old is not None:
            ret_disc = np.empty(len(self.summ_stats))
        
        for i, ss_type in enumerate(self.summ_stats):
            if isinstance(ss_type, SummaryStatistic):
                ss, diff_ss = ss_type.compute(dist, ref_dist=self.ref_dist, dist_old=dist_old)
            else:
                ss, diff_ss = self.compute_summ_stat_old(ss_type, dist, dist_old=dist_old)
                
            ret_ss[i] = ss
            if dist_old is not None:
                ret_disc[i] = diff_ss
                
        return ret_ss, ret_disc
    
    def create_logfile(self, logfile_prefix):
        if logfile_prefix is None:
            self.logfile = None
            return
        
        time_s = time.strftime("%d_%b_%H_%M_%S")
        fname = "logs/{}_{}".format(logfile_prefix, time_s)
        
        self.logfile = open(fname, "a")
        print("Created logfile '{}_{}'".format(logfile_prefix, time_s))
        
    def log(self, message):
        if self.logfile is None:
            return
        
        self.logfile.write(message)
        self.logfile.write("\n")
        self.logfile.flush()
        
        logger.info(message)
    
    def __str__(self):
        return "{}(n_species={}, reactions={}, summ_stats={}, obs={}, sim_kwargs={}, seed={})".format(
                self.__class__.__name__, self.n_species, [ str(r) for r in self.reactions], 
                self.summ_stats, self.obs, self.sim_kwargs, self.seed)
        
class CMEModel(SimModel):
    def __init__(self, n_species, reactions, summ_stats, initial_state=None, 
                 logtrans=True, gt=None, obs=None, ref_dist=None, logfile_prefix="cme", 
                 sim_kwargs = {}, seed=None):
        super().__init__(n_species=n_species, 
                         reactions=reactions,
                         summ_stats=summ_stats,
                         obs=obs,
                         logtrans=logtrans,
                         ref_dist=ref_dist,
                         initial_state=initial_state,
                         logfile_prefix=logfile_prefix,
                         sim_kwargs=sim_kwargs,
                         seed=seed)        
        self.compute_param_idcs()
            
        self.log(str(self))
        
        self.gt = gt
        if self.gt is not None:
            assert obs is None
            self.log("Simulating with following gt: {}".format(self.gt))
            self.ref_dist = self.evaluate(self.gt)
            self.obs, _ = self.compute_summ_stats(self.ref_dist)
        elif ref_dist is not None:
            self.ref_dist = ref_dist
            self.obs, _ = self.compute_summ_stats(self.ref_dist)
        else:
            assert obs is not None
            assert len(self.obs) == len(summ_stats)
            self.log("Simulating with following obs: {}".format(self.obs))
        
    def compute_param_idcs(self):
        self.param_idcs = [ i for i, r in enumerate(self.reactions) if r.rate is None ]
        
        self.d_params = len(self.param_idcs)
        
        if self.d_params == len(self.reactions):
            raise ValueError("One reaction rate has to be specified in CMEModel")
    
    def run_part(self, part, tmax, **kwargs):
        part.run(tmax, **kwargs)
        return part.get_dist()
        
    def create_system(self, params):
        reactions = self.create_rates(params)
        
        system = cme.ReactionSystem(n_species=self.n_species, 
                                    reactions = reactions,
                                    initial_state = self.initial_state)

        return system
    
    def create_rates(self, params):
        reactions = [ copy(r) for r in self.reactions ]
        
        # Rates should be converted to positive numbers
        params_iter = iter(np.power(10., params))
        
        for i, param in zip(self.param_idcs, params_iter):
            reactions[i].rate = param
        
        return reactions
    
    def __str__(self):
        return "CMEModel(n_species={}, reactions={}, summ_stats={}, obs={}, sim_kwargs={}, seed={})".format(
                self.n_species, [ str(r) for r in self.reactions ], 
                self.summ_stats, self.obs, self.sim_kwargs, self.seed)
