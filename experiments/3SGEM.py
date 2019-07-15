import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import cme
import gpopt
import models
import pdist
import lonsta
import utils

reactions = [ cme.UniReaction(rate=None, spec=0, products=(0,1,)), 
              cme.UniReaction(rate=None, spec=0, products=(3,)),
              cme.UniReaction(rate=None, spec=3, products=(0,)),
              cme.UniReaction(rate=None, spec=1, products=(1, 2)),
              cme.UniReaction(rate=None, spec=1), 
              cme.UniReaction(rate=1, spec=2) ]
summ_stats = [ models.WassersteinDistance([1,2], weights=[10,1], conv_marg=True) ]

gt = np.log10([ 4, 0.2, 0.6, 10, 1 ])

model = models.CMEModel(n_species=4,
                        reactions = reactions,
                        summ_stats = summ_stats,
                        initial_state = [1,0,0,0],
                        gt=gt,
                        sim_kwargs = { "t_block" : 5e3, "tol" : 0.2, "disable_pbar" : False, "conv_iter" : 1, "rel_es" : True,
                                        "max_iter" : 25 })

# Search ranges
ranges = [ (-1, 1),
           (-2, 0),
           (-2, 0),
           (0, 2),
           (-1, 1) ]

# GP Kernel
# Since the kernel is fit by MLE the exact hyperparameters are not very important
kernel_glob = gpopt.NonisotropicGaussianKernel(1, h=[1, 1, 1, 1, 1])
kernel_loc = gpopt.NonisotropicGaussianKernel(1, h=[0.3, 0.3, 0.3, 0.3, 0.3])

kernel = lonsta.LoNStaKernel(kernel_glob, kernel_loc, [-2,-2,-3,-3,-3], [1,1,1,1,1])

af = gpopt.LogExpectedImprovement(jitter=0.01)
gp = gpopt.GP(d=model.d_params, kernel=kernel, obs_noise=0.03)

trainer = gpopt.Trainer(gp, init_obs=300, model=model, af=af, ranges=ranges, out="results/3SGEM.pkl")

for i in range(8):
    idx = np.argmin(gp.obs_y)
    opt_pos = gp.obs_x[idx]
    
    # Penalties ensure that the nonstationarity is centered at the optimal position
    # and that nu(x) = 1 at that point
    logprior = lonsta.LoNStaKernel.get_log_prior(gp.kernel, pos_means = opt_pos, pos_scales=0.001, a_scale=0.001)
    gp.fit_kernel(logprior=logprior, 
                  opt_kwargs = {"bounds" : gp.kernel.get_bounds(pos_bounds=trainer.ranges)})
    trainer.train(75)