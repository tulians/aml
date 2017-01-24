# aml - Machine learning library.
# Utilities module.
# Author: Julian Ailan
# ===================================

"""Provides a series of methods to simplify operations."""

# Third-party modules.
import numpy as np
import matplotlib.pyplot as plt


def to_augmented_array(data):
    """Generate the augmented data set, adding a column of '1's"""
    data = np.array(data)
    if len(data):
        number_of_samples, dimension = data.shape
        augmented_data_set = np.ones((number_of_samples, dimension + 1))
        augmented_data_set[:, :-1] = data
        return augmented_data_set, dimension
    else:
        print("ERROR: An empty input vector was received. No bias unit should"
              " be added.")


def normalize(data):
    """Normalize the input vector to the range [0, 1]"""
    return (data - np.min(data)) / float(np.max(data) - np.min(data))


def display(domain, image):
    """Displays a graph"""
    plt.plot(domain, image)
    plt.show()


def mse(A, B):
    """Mean squared errors."""
    return np.sum((A - B) ** 2)
