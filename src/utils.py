# aml - Machine learning library.
# Utilities module.
# Author: Julian Ailan
# ===================================

"""Provides a series of methods to simplify operations."""

# Third-party modules.
import numpy as np


def to_augmented_array(data):
    """Generate the augmented data set, adding a column of '1's

    Args:
        data: M-dimensional array of samples.

    Returns:
        Returns the input vector in its augmented form.
    """
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
