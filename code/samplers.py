import numpy as np
from utils import bayes_factor, is_acyclic

def sample_G0(adj_G0, residuals, sigma_u, hparams):
    n_vars = adj_G0.shape[0]
    pi_prior = hparams['informed_bernoulli']['pi_bernoulli'] 
    
    new_G0 = adj_G0.copy()
    
    # Pair of nodes iteration
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j or new_G0[i, j] == 0:
                continue # skip diagonal
            
            # removal proposal
            proposed_G0 = new_G0.copy()
            proposed_G0[i, j] = 0
            # acyclic
            if is_acyclic(proposed_G0):
                #Bayes factor 
                bf = bayes_factor(residuals, i, j, sigma_u)
                acceptance_prob = min(1, bf * ((1 - pi_prior) / pi_prior))
                
                if np.random.rand() < acceptance_prob:
                    new_G0[i, j] = 0 
            
    return new_G0

