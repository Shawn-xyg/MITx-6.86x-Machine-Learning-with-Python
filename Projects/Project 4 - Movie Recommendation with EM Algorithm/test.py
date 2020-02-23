import numpy as np
import em
import common
from scipy.special import logsumexp

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, 4, 0)
# TODO: Your code here


post, log_likelihood = em.estep(X, mixture)



print(em.mstep(X, post, mixture))