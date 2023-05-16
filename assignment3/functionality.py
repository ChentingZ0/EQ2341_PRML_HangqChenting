import numpy as np


def log_likelihood(gauss_dist, x):
    px = []
    for i, dist in enumerate(gauss_dist):
        # for one observed data x and distribution d, the probability Pr(x|d)
        px.append(dist.prob(x))
    px = np.array(px)
    return np.log(px)
