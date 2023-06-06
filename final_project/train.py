from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.GaussD import GaussD
from PattRecClasses.HMM import HMM
from utils.functionality import likelihood
import numpy as np

# define distribution and observed data
# g1 = GaussD(means=np.array([0]), stdevs=np.array([1.0]))
# g2 = GaussD(means=np.array([3]), stdevs=np.array([2.0]))
g1 = GaussD(means=[0, 1], stdevs=[1, 1], cov=np.array([[1, 0], [0, 1]]))   # Distribution for state = 1
g2 = GaussD(means=[3, 1], stdevs=[2, 4], cov=np.array([[2, 0], [0, 4]]))
g = [g1, g2]

x_Seq = np.array([[-0.2, 2.6, 1.3], [-0.2, 2.6, 1.3]])
# x_Seq = np.array([[-0.2, 2.6, 1.3]])
print(x_Seq.shape)
state_seq = np.array([0, 1, 2])

pX = likelihood(g, x_Seq)
print(pX)
pX_scaled = pX / np.max(pX, axis=0)  # normalized
print(pX_scaled)

q = np.array([1, 0])
A = np.array([[0.9, 0.1], [0.2, 0.8]])  # finite

mc = MarkovChain(q, A)
hmm = HMM(mc, g)

# forward
alpha_hat, c = mc.forward(pX_scaled)

# backward
beta_hat = mc.backward(c, pX_scaled)

# total conditional probability
gamma = [[] for t in range(len(alpha_hat))]

for j in range(len(gamma)):
    gamma[j] = [alpha_hat[j][t]*beta_hat[j][t]*c[t] for t in range(len(alpha_hat[0]))]


np.set_printoptions(precision=4)
print("\nx =", x_Seq)
print("\nq =", q)
print("\nA = \n", A)
print("\nalpha_hat =\n", np.array(alpha_hat).T)
print("\nc =", np.array(c))
print("\nbeta_hat =\n", np.array(beta_hat).T)
print("\ngamma =\n", np.array(gamma).T)

log_prob = hmm.logprob(pX)
print("\nP(X = x|λ)=", log_prob)

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

for t in range(len(epsilon)-1):
    for i in range(num_state):
        for j in range(num_state):
            epsilon[t][i, j] = alpha_hat[i, t]*A[i, j]*pX_scaled[j, t+1]*beta_hat[j, t+1]
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



