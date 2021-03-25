import numpy as np


def data_generation(n_bins, n_obs, change_prob=False, multiple_features=False):

    '''

    Generate a n_bins series of length n_obs Bernoulli trials.
    For each bin the probability p of 1 changes over time, and 
    the probability that it changes depends on prob_p_change.
    multiple_features is a boolean parameter that controls the 
    addition of other features, for example they can represent the
    expected rubbish for that bin (evaluated as the average observed
    rubbish at all the previous times) or other aggregate informations.

    '''

    if multiple_features:
        data = np.empty(shape=(n_bins, n_obs, 2))
    else:
        data = np.empty(shape=(n_bins, n_obs, 1))
    
    for j in range(n_bins):

        prob_p_change = np.random.rand() if change_prob else 0
        p = np.random.rand()

        
        for i in range(n_obs):
            r = np.random.rand()
            p = np.random.rand() if r < prob_p_change else p
            
            if multiple_features:
                temp = []
                temp.append(np.random.binomial(n=1, p=p, size=1)[0])
                temp = temp + [1]
                data[j, i] = temp
            else:
                data[j, i] = np.random.binomial(n=1, p=p, size=1)

    return data