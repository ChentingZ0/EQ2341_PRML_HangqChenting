from PattRecClasses.MarkovChain import MarkovChain
from PattRecClasses.GaussD import GaussD
from PattRecClasses.HMM import HMM
from functionality import likelihood
import numpy as np

# define distribution and observed data
g1 = GaussD(means=np.array([0]), stdevs=np.array([1.0]))
g2 = GaussD(means=np.array([3]), stdevs=np.array([2.0]))
g = [g1, g2]
x_Seq = np.array([-0.2, 2.6, 1.3])

pX = likelihood(g, x_Seq)
pX_scaled = pX / np.max(pX, axis=0)  # normalized

q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
mc = MarkovChain(q, A)
h = HMM(mc, g)
alpha_hat, norms = mc.forward(pX_scaled)

np.set_printoptions(precision=3)
print("\nx =", x_Seq)
print("\nq =", q)
print("\nA = \n", A)
print("\nalpha_hat =\n", np.array(alpha_hat).T)
print("\nc =", np.array(norms))

log_prob = h.logprob(pX)
print("\nP(X = x|λ)=", log_prob)

