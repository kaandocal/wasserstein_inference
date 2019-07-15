import numpy as np

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

try:
    import pdist
except ModuleNotFoundError:
    logger.warning("Could not load module pdist")
    
dtype = np.double

class Reaction:
    """ Reaction base class 

        Every reaction has the following parameters:

        - rate: float
            Base rate of the reaction

        - dist: float (default: 0)
            Distance from epicenter at which products appear

        - products: array or None (default: None)
            List of products created during the reaction. Entries must be of the following forms:
            * spec: int 
                Denotes one reactant of the given species 
            * (b: float, spec: int)
                Denotes a burst of the given species following the geometric distribution with mean b

    """

    def __init__(self, rate, dist = 0, products = None):
        self.rate = rate
        self.dist = dist
        self.products = products

class GenReaction(Reaction):
    """
        The GenReaction class represents reactions without educts.
    """

    def __init__(self, rate, dist = 0, products = None):
        super().__init__(rate, dist, products)
        
    def __str__(self):
        return "GenReaction(rate={}, dist={}, products={})".format(self.rate, self.dist, self.products)
        
class UniReaction(Reaction):
    """ The UniReaction class represents unimolecular reactions. 
        
        Parameters:
        - spec: int
            The species undergoing the reaction
    """

    def __init__(self, rate, spec, dist = 0, products = None):
        super().__init__(rate, dist, products)
        self.spec = spec
        
    def __str__(self):
        return "UniReaction(rate={}, spec={}, dist={}, products={})".format(self.rate, self.spec, self.dist, self.products)
        
class BiReaction(Reaction):
    """
        The BiReaction class represents bimolecular reactions. Note that for reactions 
        of two particles of the same species every combination of two particles is only counted once.

        Parameters:
        - specA, specB: int
            The species of the two particles undergoing the reaction
    """

    def __init__(self, rate, specA, specB, dist = 0, products = None):
        super().__init__(rate, dist, products)
        self.specA = specA
        self.specB = specB
        
    def __str__(self):
        return "BiReaction(rate={}, specA={}, specB={}, dist={}, products={})".format(self.rate, self.specA, self.specB, self.dist, self.products)
    
class ReactionSystem:
    """ This class stores information about the model in a CME reaction system.
    
        Arguments:
            n_species: int
                Number of species in the system. Species are 
                enumerated starting from 0.
               
            reactions: array of reactions (default: ())
                List of allowed reactions in the system
            
            initial_state: array of ints
                Initial number of particles per species. 
                
        Attributes:
            block_size: int (default: 100)
                Size of block in particle buffer (internal)
    """
    def __init__(self, n_species, reactions = (), initial_state = None):
        self.n_species = n_species
        
        if initial_state is None:
            initial_state = np.zeros(n_species, dtype=int)
        
        self.initial_state = initial_state
        
        self.reactions = reactions
        self.sort_reactions(reactions)
        
    def sort_reactions(self, reactions):
        """ Sort reactions by type during initialisation """
        self.reactions_gen = []
        self.reactions_uni = []
        self.reactions_bi = []
        
        for reac in reactions:
            if isinstance(reac, GenReaction):
                self.reactions_gen.append(reac)
            elif isinstance(reac, UniReaction):
                self.reactions_uni.append(reac)
            elif isinstance(reac, BiReaction):
                self.reactions_bi.append(reac)
            else:
                raise TypeError("Unknown reaction type for '{}'".format(reac))
        
    def create_particle_system(self, seed=None):
        """ Create a new instance of ParticleSystem associated to this reaction system """
        return ParticleSystem(self, seed=seed)
    
class ParticleSystem:
    """ This class represents the state of a particle system associated to a ReactionSystem 
    
        Args:
            system: ReactionSystem
                Associated reaction system defining the model to use
                
            seed: int or None (default: None)
                Optional seed for random number generator
                
        Attributes:
            counts: n_species array of ints
                Per-species particle counts
                
            t: float
                Current time in the system
                
            events: array of events
                Ordered list of time-stamped events in the system
    """
    def __init__(self, system, seed=None):
        self.system = system
        
        self.counts = np.zeros(system.n_species, dtype=int)
        
        self.rng = np.random.RandomState(seed=seed)
       
        # Precompute some things for ease of access during simulations
        self.gen_rates = np.asarray([ r.rate for r in self.system.reactions_gen ])
        
        self.uni_rates = np.asarray([ r.rate for r in self.system.reactions_uni ])
        self.uni_spec = np.asarray([ r.spec for r in self.system.reactions_uni ], dtype=int)
       
        self.t = 0
        self.events = []
        
        # Used for fast updating of particle count trajectories 
        self.events_iter = None
        
        self.add_initial_molecules()
        
    ### INTERNALS ###
    def add_products(self, products):
        """ Add list of particles to the reaction system. Returns list of particles placed.
        """

        # Create list of products to be placed
        products = self.expand_products(products)
        
        log = []
        for product in products:
            self.counts[product] += 1
            
            log.append((product,))
            
        return log
    
    def expand_products(self, products):
        """ 
            Convert description of reaction products into list of products to be
            added, drawing particle numbers for products produced in bursts.
        """
        ret = []
        for prod in products:
            if type(prod) == int:
                ret.append(prod)
            else:
                m, spec = prod
                
                p = 1 / m
                n = self.rng.geometric(p) - 1
                ret += [ spec for i in range(n) ]
                
        return ret
    
    def add_initial_molecules(self):
        """ Place initial molecules in the system """
        assert self.t == 0
        
        for spec, n_init in enumerate(self.system.initial_state):
            product_log = self.add_products([ spec for i in range(n_init) ])
                
            event = ("gen", None, product_log)
            self.events.append((0, event))
            
    def compute_bi_rates(self):
        """ Compute reaction rates for bimolecular reactions """
        rates = np.empty(len(self.system.reactions_bi),)
        
        for i, reac in enumerate(self.system.reactions_bi):
            # Do not possible overcount reactant pairs if both educts are of the same species
            if reac.specA == reac.specB:
                combs = 0.5 * self.counts[reac.specA] * (self.counts[reac.specA] - 1)
            else:
                combs = self.counts[reac.specA] * self.counts[reac.specB]
                
            rates[i] = reac.rate * combs
            
        return rates
    
    ### UPDATES ###
    def perform_gen_reaction(self, reaction):
        """ Simulate one occurrence of the specified generative reaction """
        product_log = self.add_products(reaction.products)
        return ("gen", reaction, product_log)
                        
    def perform_uni_reaction(self, reaction):
        """ Simulate one occurrence of the specified unimolecular reaction """
        self.counts[reaction.spec] -= 1
        product_log = ()
        
        if reaction.products is not None:
            product_log = self.add_products(reaction.products)
            
        return ("uni", reaction, product_log)
    
    def perform_bi_reaction(self, reaction, rates):
        """ Simulate one occurrence of the specified bimolecular reaction.
            Refer to perform_uni_reaction for more information. """
        
        self.counts[reaction.specA] -= 1
        self.counts[reaction.specB] -= 1
        
        product_log = ()
        if reaction.products is not None:
            product_log = self.add_products(reaction.products)
            
        return ("bi", reaction, product_log)
        
    ### MAIN LOOP ###
    def run(self, tmax, disable_pbar=True):
        """ Run simulation for tmax time units using the Gillespie algorithm """
        t0 = self.t
        
        gen_rates = self.gen_rates
        
        with tqdm(total=tmax, 
                  desc="Time simulated: ", 
                  unit="tu", 
                  disable=disable_pbar) as pbar:
            while True:
                uni_rates = self.uni_rates * self.counts[self.uni_spec]
                bi_rates = self.compute_bi_rates()
                rate = np.sum(gen_rates) + np.sum(uni_rates) + np.sum(bi_rates)
                
                # Nothing happening
                if rate == 0.0 or not np.isfinite(rate):
                    if not np.isfinite(rate):
                        logger.warning("Numerical blow-up in CME simulation")
                       
                    # Pretend the last reaction happened at time tmax.
                    # This is necessary for updating progress bar correctly.
                    dt = 0
                    break

                dt = self.rng.exponential(1 / rate)
                self.t += dt
                if self.t > t0 + tmax:
                    break
                    
                pbar.update(dt)

                # The Gillespie algorithm randomly samples a possible event
                # with probabilities proportional to the rates
                p = self.rng.uniform(0, rate)

                # Zero-molecular reaction happening
                if p <= np.sum(gen_rates):
                    for reac, rate in zip(self.system.reactions_gen, gen_rates):
                        if p >= rate:
                            p -= rate
                            continue

                        event = self.perform_gen_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Unimolecular reaction happening
                elif p <= np.sum(gen_rates) + np.sum(uni_rates): 
                    p -= np.sum(gen_rates)
                    
                    for reac in self.system.reactions_uni:
                        if p >= reac.rate * self.counts[reac.spec]:
                            p -= reac.rate * self.counts[reac.spec]
                            continue

                        event = self.perform_uni_reaction(reac)
                        self.events.append((self.t, event))
                        break

                # Bimolecular reaction happening
                else:
                    p -= np.sum(gen_rates) + np.sum(uni_rates)
                    
                    for reac, rates in zip(self.system.reactions_bi, bi_rates):
                        if p >= np.sum(rates):
                            p -= np.sum(rates)
                            continue
                            
                        event = self.perform_bi_reaction(reac, rates)
                        self.events.append((self.t, event))
                        break
           
            # This set sprogress bar value to t0 + tmax
            pbar.update(dt - (self.t - t0 - tmax))
            self.t = t0 + tmax
    
    ### UTILITY FUNCTIONS ###
    def update_count_species(self, **kwargs):
        return self.count_species(**kwargs)
    
    def get_dist(self, t_max=None):
        """ 
            Return the distribution of particle numbers over the lifetime of the system
            using time-averaging.
        """
        counts = np.zeros((self.system.n_species, len(self.events) + 1), dtype=int)
        weights = np.zeros(len(self.events) + 1)

        if t_max is None:
            t_max = self.t
            
        t_last = 0
        i = 0
        
        for t, e in self.events:
            if e[0] == "jump" or t > t_max:
                continue

            weights[i] = t - t_last
            t_last = t

            i += 1
            counts[:,i] = counts[:,i-1]
            if e[0] == "gen":
                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1
            if e[0] == "uni":
                reac = e[1]
                counts[reac.spec][i] -= 1

                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1
            elif e[0] == "bi":
                reac = e[1]

                counts[reac.specA][i] -= 1
                counts[reac.specB][i] -= 1

                product_log = e[2]
                for spec_product in product_log:
                    counts[spec_product][i] += 1

        weights[i] = t_max - t_last

        return pdist.ParticleDistribution(counts[:,:i+1], weights=weights[:i+1])
