""" 

cme_julia - Python binding for CME simulator written in Julia

This file can be used similarly to the original file cme.py;
the Reaction, ReactionSystem and ParticleSystem classes are defined
the same way and expose a subset of the Python API. See the file
cme.jl for information on the Julia implementation.

"""

import pdist
import numpy as np

from julia.api import Julia
jul = Julia(compiled_modules=False)

with open("cme.jl") as inpf:
    code = inpf.read()
    
jul.eval(code)

class Reaction:
    def __init__(self, rate, products = None):
        self.rate = rate
        self.products = products

class GenReaction(Reaction):
    def __init__(self, rate, products = None):
        super().__init__(rate, products)
        
    def __str__(self):
        return "GenReaction(rate={}, products={})".format(self.rate, self.products)
        
class UniReaction(Reaction):
    def __init__(self, rate, spec, products = None):
        super().__init__(rate, products)
        self.spec = spec
        
    def __str__(self):
        return "UniReaction(rate={}, spec={}, products={})".format(self.rate, self.spec, self.products)
        
class BiReaction(Reaction):
    def __init__(self, rate, specA, specB, products = None):
        super().__init__(rate, products)
        self.specA = specA
        self.specB = specB
        
    def __str__(self):
        return "BiReaction(rate={}, specA={}, specB={}, products={})".format(self.rate, self.specA, self.specB, self.products)

class ReactionSystem:
    def __init__(self, n_species, reactions = (), initial_state = None):
        j_reactions = []
        j_GenReaction = jul.eval("GenReaction")
        j_UniReaction = jul.eval("UniReaction")
        j_BiReaction = jul.eval("BiReaction")
        
        for reaction in reactions:
            if reaction.products is None:
                products = []
            else:
                products = list(reaction.products)
            
            for i, product in enumerate(products):
                if not isinstance(product, int):
                    products[i] = (float(product[0]), int(product[1]))
                    
            if isinstance(reaction, GenReaction):
                j_reaction = j_GenReaction(float(reaction.rate), products)
            elif isinstance(reaction, UniReaction):
                j_reaction = j_UniReaction(float(reaction.rate), int(reaction.spec), products)
            elif isinstance(reaction, BiReaction):
                j_reaction = j_BiReaction(float(reaction.rate), int(reaction.specA), int(reaction.specB), products)
            else:
                raise NotImplementedError()
                
            j_reactions.append(j_reaction)
            
        j_ReactionSystem = jul.eval("ReactionSystem")
        
        if initial_state is None:
            initial_state = np.zeros(self.n_species, dtype=int)
        self.n_species = int(n_species)
        self.reactions = reactions
        self.initial_state = initial_state
        
        self._jul = jul
        self._jobj = j_ReactionSystem(int(n_species), j_reactions, initial_state)
        
    def create_particle_system(self, seed=None):
        return ParticleSystem(self, seed=seed)
        
# Auxiliary functions that allow Julia to access tqdm progree bars
def pbar_create(total, desc, unit):
    return tqdm.tqdm(total=total, desc=desc, unit=unit)

def pbar_update(pbar, dt):
    pbar.update(dt)

def pbar_close(pbar):
    pbar.close()

class ParticleSystem:
    def __init__(self, system, seed=None):
        self.system = system
        self.seed = seed
        
        self._jul = system._jul
        j_ParticleSystem = self._jul.eval("ParticleSystem")
        self._jrun = self._jul.eval("run")
        self._jobj = j_ParticleSystem(system._jobj, seed)
        self.last_state = None
        
    def get_dist(self):
        
        j_get_dist_data = self._jul.eval("get_dist_data")
        
        if self.last_state is not None:
            data = j_get_dist_data(self._jobj, *self.last_state)
        else:
            data = j_get_dist_data(self._jobj)

        #self.last_state = (data, np.copy(self._jobj.cells), self._jobj.t, len(self._jobj.events))
        
        ret = pdist.ParticleDistribution(data, hist=True)
        return ret
    
    def run(self, t_max, disable_pbar=True):
        self._jrun(self._jobj, float(t_max), disable_pbar=disable_pbar,
                   pbar_create=pbar_create, pbar_update=pbar_update,
                   pbar_close=pbar_close)
    
        
