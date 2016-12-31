# aml - Machine learning library.
# Utilities module.
# Author: Julian Ailan
# ===================================

# Built-in modules.
import json
from pprint import pprint
# Third-party modules.
import numpy as np

"""Provides a series of methods to simplify simple tasks."""

import numpy as np


def to_augmented_array(data):
    """Generate the augmented data set, adding a column of '1's

    Args:
        data: M-dimensional array of samples.

    Returns:
        Returns the input vector in its augmented form.
    """
    data = np.array(data)
    if data.ndim is 1:
        number_of_samples = 1
        dimension = data.size
    elif data.ndim is 2:
        number_of_samples, dimension = data.shape
    else:
        # TODO: Define an exception
        return False
    augmented_data_set = np.ones((number_of_samples, dimension + 1))
    augmented_data_set[:, :-1] = data
    return augmented_data_set, dimension


def pending_tasks():
    # TODO: Print only those that are still left to complete.
    with open("../pending.json") as tasks_file:
        data = json.load(tasks_file)
    pprint(data)
