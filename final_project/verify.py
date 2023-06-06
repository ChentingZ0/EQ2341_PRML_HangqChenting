from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.GaussD import GaussD
from PattRecClasses.HMM import HMM
from utils.functionality import likelihood
import numpy as np

# define distribution and observed data
g1 = GaussD(means=np.array([0]), stdevs=np.array([1.0]))
g2 = GaussD(means=np.array([3]), stdevs=np.array([2.0]))
g = [g1, g2]
x_Seq = np.array([-0.2, 2.6, 1.3])

pX = likelihood(g, x_Seq)
pX_scaled = pX / np.max(pX, axis=0)  # normalized

q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])  # finite
mc = MarkovChain(q, A)
hmm = HMM(mc, g)

# forward
alpha_hat, c = mc.forward(pX_scaled)

# backward
beta_hat = mc.backward(c, pX_scaled)


np.set_printoptions(precision=4)
print("\nx =", x_Seq)
print("\nq =", q)
print("\nA = \n", A)
print("\nalpha_hat =\n", np.array(alpha_hat).T)
print("\nc =", np.array(c))
print("\nbeta_hat =\n", np.array(beta_hat).T)
log_prob = hmm.logprob(pX)
print("\nP(X = x|λ)=", log_prob)

