# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
# ===================================

"""Provides an abstraction to generate neural networks."""

# Project's modules
import utils as u
import threshold as th
# Third-party modules
import numpy as np


class NeuralNetwork(object):
    """Layer against layer neural network architecture."""

    def __init__(self, layers=[2, 2, 1], activation_function="logistic"):
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
        self.weights = self._generate_weights(layers)

    def _generate_weights(self, layers):
        """Generates the network's synaptic weights. Bias weights are
        respectively added to the output list.

        Args:
            layers: list which includes the number of units in each layer.

        Returns:
            weights: list of synaptic weights.
        """
        weights = []
        for i in range(1, len(layers) - 1):
            weights.append(
                2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1)
        weights.append(
            2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1)
        return weights

    def feedforward(self, sample):
        """Computes the output of the network given a sample vector.

        Args:
            sample: input to the network.

        Returns:
            output: vector with output layer values.
        """
        output, _ = u.to_augmented_array(sample)
        for weight in xrange(len(self.weights)):
            output = self.activation_function(
                np.dot(output, self.weights[weight]))
        return output

    def train(self, training_samples, labels, learning_rate=0.1,
              epochs=10000):
        """Trains the network using stochastic gradient descent (SGD).

        Args:
            training_samples: list of samples used to train the network's
            weights.
            labels: outputs associated to the training_samples.
            learning_rate: 'speed' at which the SGD algorithm learns.
            epochs: number of iterations to perform in SGD.

        Returns:
            No data is returned.
        """
        # Format vector inputs.
        training_samples, _ = u.to_augmented_array(training_samples)
        labels = np.array(labels)

        for epoch in xrange(epochs):
            sample_index = np.random.randint(training_samples.shape[0])
            activations = [training_samples[sample_index]]
            # Forward pass.
            for weight in xrange(len(self.weights)):
                activations.append(
                    self.activation_function(
                        np.dot(activations[weight], self.weights[weight])))
            # Backpropagation starts:
            # 1- Output layer weights compensation.
            dEtotal_wrt_dOutput = labels[sample_index] - activations[-1]
            dOutput_wrt_dInput = self.activation_derivative(activations[-1])
            deltas = [dEtotal_wrt_dOutput * dOutput_wrt_dInput]
            # 2- Hidden layers weights compensation.
            for layer in xrange(len(activations) - 2, 0, -1):
                deltas.append(
                    deltas[-1].dot(self.weights[layer].T) *
                    self.activation_derivative(activations[layer]))
            deltas.reverse()
            # 3- Weights update.
            for index in xrange(len(self.weights)):
                layer = np.atleast_2d(activations[index])
                delta = np.atleast_2d(deltas[index])
                self.weights[index] += learning_rate * np.dot(layer.T, delta)
