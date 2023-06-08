from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.GaussD import GaussD
from PattRecClasses.HMM import HMM
from utils.functionality import likelihood
import numpy as np


# define distribution and observed data
g1 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
g2 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
g3 = GaussD(means=[0, 0, 0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
g = [g1, g2, g3]


test_data = np.load('./data/test_data.npy').T

q = np.array([1, 0, 0])
A = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])  # finite


def hmm_test(q, A, g, x):
    """HMM test function using EM algorithm
        Input parameters:
            hmm: an object created from the class HMM
            q: the initial state probabilities, len(q) = num_state
            A: transition matrix of hmm, len(q) = A.shape[0]
            g: the output distribution list, each element of g should be an object created from
                the class GaussD. The length of g len(g) should equal to the number of State
                that is, len(g) = len(q) = A.shape[0]
            x: observed sequence of shape [data dimensions, number of samples: T]
                data dimensions = mean.shape[0] = cov.shape[0], mean and cov are the mean and covariance
                matrix of the GaussianD objects.

        Output parameters:
            gamma: all conditional state sequence
            state_seq: output state sequence
    """

    mc = MarkovChain(q, A)
    pX = likelihood(g, x)
    # print(pX)
    pX_scaled = pX / np.max(pX, axis=0)  # normalized
    # print(pX_scaled)

    # forward
    alpha_hat, c = mc.forward(pX_scaled)

    # backward
    beta_hat = mc.backward(c, pX_scaled)

    # total conditional probability
    gamma = [[] for t in range(len(alpha_hat))]

    for j in range(len(gamma)):
        gamma[j] = [alpha_hat[j][t] * beta_hat[j][t] * c[t] for t in range(len(alpha_hat[0]))]

    state_seq = [state_t.index(max(state_t)) for state_t in gamma]
    gamma = np.array(gamma)

    return state_seq, gamma


# testing
print(test_data.shape)
state_seq, gamma = hmm_test(q, A, g, test_data)

