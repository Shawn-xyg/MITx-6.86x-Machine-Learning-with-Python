import numpy as np
import kmeans
import common
import naive_em
import em
import scipy.stats

X = np.loadtxt("toy_data.txt")

# K-means: determining the centroids by comparing the cost
k_dict = dict()
total_cost_dict = dict()
for seed in range(5):
    total_cost = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        cost = kmeans.run(X, mixture, post)[2]
        total_cost += cost
        k_dict.update({(seed, k): cost})
    total_cost_dict.update({seed: total_cost})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = total_cost_dict[0]
for k, v in total_cost_dict.items():
    if v < optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in total_cost_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k

## Best k size
# Get the lowest cost
optimal_k_cost = k_dict[(optimal_seed, 1)]
# Create a new dictionary for k size
optimal_k_dict = dict()
for i in range(1, 5):
    optimal_k_dict.update({(optimal_seed, i): k_dict[(optimal_seed, i)]})
for k, v in optimal_k_dict.items():
    if v < optimal_k_cost:
        optimal_k_cost = v
    else:
        optimal_k_cost = optimal_k_cost
# Get the seed associated with the lowest cost
for k, v in optimal_k_dict.items():
    if v == optimal_k_cost:
        optimal_k = k[1]

### Plotting the k clusters
optimal_seed_k = list()
optimal_seed_k_post = list()
title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, cost = kmeans.run(X, initial_mixture, initial_post)
    optimal_seed_k.append(mixture)
    optimal_seed_k_post.append(post)
    title_list.append(("K-means: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, optimal_seed_k[i], optimal_seed_k_post[i], title_list[i])

####### Compare k-means with EM

# K-means: determining the centroids by comparing the cost
em_k_dict = dict()
em_total_likelihood_dict = dict()
for seed in range(5):
    em_total_likelihood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        likelihood = naive_em.run(X, mixture, post)[2]
        em_total_likelihood += likelihood
        em_k_dict.update({(seed, k): likelihood})
    em_total_likelihood_dict.update({seed: em_total_likelihood})

### get the best seed and the best k size that minimizes the cost

## Best seed
# Get the lowest cost
optimal_seed_cost = em_total_likelihood_dict[0]
for k, v in em_total_likelihood_dict.items():
    if v > optimal_seed_cost:
        optimal_seed_cost = v
    else:
        optimal_seed_cost = optimal_seed_cost
# Get the seed associated with the lowest cost
for k, v in em_total_likelihood_dict.items():
    if v == optimal_seed_cost:
        optimal_seed = k
print(em_k_dict)

### Plotting the k clusters
em_optimal_seed_k = list()
em_optimal_seed_k_post = list()
em_title_list = list()
for i in range(1, 5):
    initial_mixture, initial_post = common.init(X, i, seed = optimal_seed)
    mixture, post, likelihood = naive_em.run(X, initial_mixture, initial_post)
    em_optimal_seed_k.append(mixture)
    em_optimal_seed_k_post.append(post)
    em_title_list.append(("Gaussian Mixture: The mixture plot when k = {}".format(i)))

for i in range(4):
    common.plot(X, em_optimal_seed_k[i], em_optimal_seed_k_post[i], title_list[i])

### Seed with best BIC
BIC_k_dict = dict()
BIC_total_likelihood_dict = dict()
for seed in range(5):
    BIC_total_likehood = 0
    for k in range(1, 5):
        mixture, post = common.init(X=X, K=k, seed=seed)
        log_likelihood = naive_em.run(X, mixture, post)[2]
        BIC = common.bic(X, mixture, log_likelihood)
        BIC_total_likehood += BIC
        BIC_k_dict.update({(seed, k): BIC})
    total_cost_dict.update({seed: BIC_total_likehood})
print(BIC_k_dict)


### Determining the initialization



### run EM algorithm on X


