# aml - Machine learning library.
# Perceptron module.
# Author: Julian Ailan
#===================================

"""Provides different implementations of perceptrons."""

import numpy as np
import threshold as th

class Perceptron(object):
    """Classic perceptron implementation."""

    def __init__(self, input_weights = None, learning_factor=1, epochs=50,
                 activation_function="sigmod"):
        self.w = np.array(input_weights)
        self.learning_factor = learning_factor
        self.epochs = epochs
        if activation_function == "unitstep":
            self.activation_function = th.UnitStep()
        else:
            self.activation_function = th.Sigmoid()


    def train(self, data_set, labels, alternate_number_of_epochs=None):
        """Computes the components of the weights vector 'w'.

        Args:
            data_set: Array of M-dimensional samples.
            labels: Array of labels that represent the class of each sample.

        Returns:
            w: Weights vector, containing the parameters of the hyperplane.
        """

        # Added to increase flexibility when testing with different epochs
        # values, without the need of defining new perceptrons.
        if alternate_epochs:
            self.epochs = alternate_number_of_epochs

        data_set = np.array(data_set)
        number_of_samples, dimension = data_set.shape

        # Generate the augmented data set, adding a column of '1's
        augmented_data_set = np.ones((number_of_samples, dimension + 1))
        augmented_data_set[:,:-1] = data_set

        self.w = 1e-6 * np.random.rand(dimension + 1)

        for _ in xrange(self.epochs):
            for sample, target in zip(augmented_data_set, labels):
                predicted_output = self.activation_function(np.dot(self.w,
                                                                   sample))
                update = (self.learning_factor * (target - predicted_output) *
                          sample)
                self.w += update


    def output(self, input_sample, alternate_activation_function=None):
        """Compute the output of the perceptron for a given sample input.

        Args:
            input_sample: Input vector containing samples values.

        Returns:
            Returns the output of the activation function
        """
        input_sample = np.array(input_sample)
        length = input_sample.size
        augmented_sample = np.ones((1, length + 1))
        augmented_sample[:,:-1] = input_sample

        if alternate_activation_function == "raw":
            return np.dot(self.w, augmented_sample.T)
        elif alternate_activation_function:
            return alternate_activation_function(np.dot(self.w,
                                                        augmented_sample.T))
        else:
            return self.activation_function(np.dot(self.w, augmented_sample.T))
