import numpy as np


def likelihood(gauss_dist, x):
    px = []
    for i, dist in enumerate(gauss_dist):
        # for one observed data x and distribution d, the probability Pr(x|d)
        px_i = []
        for t in range(x.shape[1]):
            # print(x[:,t])
            px_i.append(dist.prob(x[:, t]))
        px.append(px_i)
    px = np.array(px)
    return px
