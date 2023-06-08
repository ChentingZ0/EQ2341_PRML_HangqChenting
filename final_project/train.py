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


train_data = np.load('./data/train_data.npy').T

q = np.array([1, 0, 0])
A = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])  # finite


def hmm_train(q, A, g, x):
    """HMM training function using EM algorithm
        Input parameters:
            mc: an object created from the class Markov Chain
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
            q_new: updated q
            A_new: updated A
            mean_new: list of new mean matrix for the updated Gaussian distributions
            cov_new: list of new covariance matrix for the  updated Gaussian distributions"""

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

    # training
    # update q
    q_new = [gamma[0][i] for i in range(len(gamma[0]))]  # eq 7.55
    print("q_new = \n", q_new)

    # update A
    epsilon = [np.zeros(A.shape) for t in range(len(alpha_hat))]
    alpha_hat = np.array(alpha_hat).T
    beta_hat = np.array(beta_hat).T
    num_state = A.shape[0]
    epsilon_bar = np.zeros(epsilon[0].shape)

    for t in range(len(epsilon) - 1):
        for i in range(num_state):
            for j in range(num_state):
                epsilon[t][i, j] = alpha_hat[i, t] * A[i, j] * pX_scaled[j, t + 1] * beta_hat[j, t + 1]
                # eq 6.19
        epsilon_bar += epsilon[t]

    epsilon_sum = np.sum(epsilon_bar)  # eq 6.15 and eq 6.16
    A_new = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_new[i, j] = epsilon_bar[i, j] / np.sum(epsilon_bar[i, :])  # eq 6.13

    print("A = \n", A)
    print("A_new = \n", A_new)

    # update B
    gamma = np.array(gamma).T
    mean_new = []
    for i in range(len(g)):
        numerator = [gamma[i, t] * x[:, t] for t in range(x.shape[1])]
        numerator = np.sum(numerator, axis=0)
        mu_new_i = numerator / np.sum(gamma[i, :])
        mean_new.append(mu_new_i)

    print(mean_new)

    cov_new = []
    for i in range(len(g)):
        numerator = None
        for t in range(x.shape[1]):
            # print((x_Seq[:, t] - mean_new[i]).T @ (x_Seq[:, t] - mean_new[i]))
            res1 = np.expand_dims((x[:, t] - mean_new[i]), axis=1)
            res_product = res1 @ res1.T
            numerator = gamma[i, t] * res_product if t == 0 else numerator + gamma[i, t] * res_product
        cov_new.append(numerator / np.sum(gamma[i, :]))

    print("covariance new = \n", cov_new)

    return q_new, A_new, mean_new, cov_new


# training
q_i, A_i, mean_i, cov_i = None, None, None, None
for iteration in range(10):
    print(iteration)
    if iteration == 0:
        q_i, A_i, mean_i, cov_i = hmm_train(q, A, g, train_data)
    else:
        # g1 = GaussD(means=mean_i[0], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # g2 = GaussD(means=mean_i[1], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # g3 = GaussD(means=mean_i[2], cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        g1 = GaussD(means=mean_i[0], cov=cov_i[0])
        g2 = GaussD(means=mean_i[1], cov=cov_i[1])
        g3 = GaussD(means=mean_i[2], cov=cov_i[2])

        g_i = [g1, g2, g3]
        q_i, A_i, mean_i, cov_i = hmm_train(q_i, A_i, g_i, train_data)

