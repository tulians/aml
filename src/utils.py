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
    if not isinstance(data, list):
        data = [data]
    data = np.array(data)
    if data.size > 0:
        if data.ndim is 1:
            number_of_samples = 1
            dimension = data.size
        elif data.ndim is 2:
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
