# Neural network module.
# ===================================

"""Provides an abstraction to generate neural networks."""

# Project's modules
import utils as u
import threshold as th
from training_methods import backpropagation
# Third-party modules
import numpy as np


class FeedforwardNeuralNetwork(object):
    """Fully connected neural network architecture."""

    def __init__(self, layers=[2, 2, 1], activation_function="bentidentity"):
        """Neural network class constructor.
        Args:
            layers: list which includes the number of units in each layer. The
            first item on the list corresponds to the number of units in the
            input layer. The last element corresponds to the number of units
            in the output layer. Thus, the numbers in between correspond to
            the units in the hidden layers.
            activation_function: sigmoid function to use as unit activation.
        Returns:
            No data is returned.
        """
        self.layers = layers
        self.activation_function = th.activation_functions[activation_function]
        self.activation_derivative = th.activation_derivatives[
            activation_function]
        self.weights = self._generate_weights()

    def _generate_weights(self):
        """Generates the network's synaptic weights. Bias weights are
        respectively added to the output list.
        Args:
            No input arguments.
        Returns:
            weights: list of synaptic weights.
        """
        weights = []
        for i in range(1, len(self.layers) - 1):
            weights.append(2 * np.random.random(
                (self.layers[i - 1] + 1, self.layers[i] + 1)) - 1)
        weights.append(2 * np.random.random(
            (self.layers[i] + 1, self.layers[i + 1])) - 1)
        return weights

    def _feedforward(self, sample):
        """Computes the output of the network given a sample vector.
        Args:
            sample: input to the network.
        Returns:
            output: vector with output layer values.
        """
        output = sample
        for weight in xrange(len(self.weights)):
            output = self.activation_function(
                np.dot(output, self.weights[weight]))
        return output

    def train(self, training_samples, labels, method="SGD",
              cost_function=u.mse, learning_rate=0.1, epochs=100000,
              ret_error=True, tolerance=1e-10, display=u.display):
        # Format vector inputs and outputs
        training_samples, _ = u.to_augmented_array(training_samples)
        labels = np.array(labels)

        training_method = backpropagation.methods[method]
        return training_method(
            self.layers, self.weights, self.activation_function,
            self.activation_derivative, training_samples, labels,
            self._feedforward, cost_function, learning_rate, epochs,
            ret_error, tolerance, display
        )

    def predict(self, samples):
        """Computes the output of the trained network given a dataset.
        Args:
            samples: data to compute the output from.
        Returns:
            output: predicted outputs for given dataset.
        """
        output = []
        samples, _ = u.to_augmented_array(samples)
        for sample in samples:
            output.append(self._feedforward(sample))
        return output
