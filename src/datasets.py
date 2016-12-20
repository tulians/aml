# aml - Machine learning library.
# Data generation module.
# Author: Julian Ailan
# ===================================

import numpy as np


def generate_std_normal_set(number_of_samples=100, dimension=2):
    dataset = []
    for _ in xrange(number_of_samples):
        dataset.append(np.random.randn(dimension))
    return np.array(dataset)


def generate_multivariate_normal_set(number_of_samples=100, mean=[0, 0],
                                     cov=[[1, 0], [0, 1]]):
    "Returns an array of 'number_of_samples' samples, one per row"
    return np.random.multivariate_normal(mean, cov, number_of_samples)
