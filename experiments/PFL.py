import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import cme
import gpopt
import models
import pdist
import lonsta
import utils

b = 3

reactions = [ cme.UniReaction(rate=1, spec=0), 
              cme.UniReaction(rate=None, spec=1, products=(1, (b, 0))), 
              cme.UniReaction(rate=70, spec=2, products=(2, (b, 0))),
              cme.UniReaction(rate=None, spec=2, products=(1,0)),
              cme.BiReaction(rate=None, specA=1, specB=0, products=(2,)) ]

summ_stats = [ models.WassersteinDistance([0]) ]
initial_state = [0, 1, 0]

gt = np.log10([ 0.2, 400, 2.5 ])

model = models.CMEModel(n_species=3,
                        reactions = reactions,
                        summ_stats = summ_stats,
                        initial_state = initial_state,
                        logtrans = True,
                        gt=gt,
                        sim_kwargs = { "t_block" : 3e3, "tol" : 0.01, "disable_pbar" : False, "conv_iter" : 1, "rel_es" : True })

ranges = [ (-1, 1),
           (1, 3),
           (-1, 1) ]

kernel = lonsta.LoNStaKernel(gpopt.NonisotropicGaussianKernel(1, h=[1,1,1]), 
                             gpopt.NonisotropicGaussianKernel(1, h=[0.1,0.1,0.1]),
                             [0,0,0], [1,1,1])

af = gpopt.LogExpectedImprovement(jitter=0.01)
gp = gpopt.GP(d=model.d_params, kernel=kernel, obs_noise=0.03)

trainer = gpopt.Trainer(gp, init_obs=75, model=model, af=af, ranges=ranges, out="results/PFL.pkl")

for i in range(6):
    idx = np.argmin(gp.obs_y)
    opt_pos = gp.obs_x[idx]
    
    logprior = lonsta.LoNStaKernel.get_log_prior(gp.kernel, pos_means = opt_pos, pos_scales=0.001, a_scale=0.001)
    gp.fit_kernel(logprior=logprior, 
                  opt_kwargs = {"bounds" : gp.kernel.get_bounds(pos_bounds=trainer.ranges)})
    trainer.train(25)