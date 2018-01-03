# Perceptron module.
# ===================================

"""Provides a traditional implementation of a perceptron."""

# Project's modules
from threshold import Threshold
from databuffer import DataBuffer


class Perceptron(object):
    """Classic perceptron implementation."""

    def __init__(self, weights=None, activation_function="logistic"):
        """Perceptron class constructor.
        -->()
            weights: Optional initial weights.
            activation_function: Sigmoid function to use in learning.
        ()-->
            None.
        """
        self.weights = weights
        self.activation_function = Threshold(activation_function)

    def train(self, data_set, labels, learning_rate=0.1, epochs=50):
        """Computes the components of the weights vector 'w'.
        -->()
            data_set: Array of M-dimensional samples.
            labels: Array of labels that represent the class of each sample.
            learning_rate: 'Speed' at which the perceptron learns.
            epochs: Number of iterations to perform in the learning algorithm.
        ()-->
            weights: Weights vector, containing the parameters of the
            hyperplane.
        """
        augmented_data_set, dimension = u.to_augmented_array(data_set)
        self.weights = 2 * np.random.rand(dimension + 1) - 1

        for _ in xrange(epochs):
            for sample, target in zip(augmented_data_set, labels):
                predicted_output = self.activation_function(
                    np.dot(self.weights, sample))
                update = (learning_rate * (target - predicted_output) *
                          sample)
                self.weights += update

    def output(self, input_sample):
        """Compute the output of the perceptron for a given sample input.
        -->()
            input_sample: Input vector containing samples values.
        ()-->
            Returns the output of the activation function.
        """

        augmented_sample, _ = np.array(u.to_augmented_array(input_sample))
        return self.activation_function(np.dot(self.weights,
                                               augmented_sample.T))
