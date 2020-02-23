import numpy as np
import common
import em
from scipy.special import logsumexp

### Collborative filtering with EM

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")
X_test = np.loadtxt("test_incomplete.txt")
X_experiment = np.loadtxt("toy_data.txt")

mixture, post = common.init(X, K=12, seed=1)

mixture, post, loglike = em.run(X, mixture, post)

X_pred = em.fill_matrix(X, mixture)

print(common.rmse(X_gold, X_pred))

print(mixture)
#print(em.fill_matrix(X_test
### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
#optimal_seed_cost = em_total_likelihood_dict[0]
#for k, v in em_total_likelihood_dict.items():
#    if v > optimal_seed_cost:
#        optimal_seed_cost = v
#    else:
#        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
#for k, v in em_total_likelihood_dict.items():
#    if v == optimal_seed_cost:
#        optimal_seed = k
#print(em_k_dict)




### Test case for exam

mixture = common.GaussianMixture(np.array([[1, 1], [1, 1]]), np.array([0.5, 0.5]), np.array([0.01, 0.99]))
post = np.ones((X_experiment.shape[0], 2)) / 2
mixture, post, loglike = em.run(X_experiment, mixture, post)

common.plot(X_experiment, mixture, post, "Test case")
print(post)