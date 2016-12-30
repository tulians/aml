# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
# ===================================

"""Provides an abstraction to generate neural networks."""

import random

import numpy as np

import threshold as th


class NeuralNetwork(object):
    """Layer against layer neural network architecture."""

    def __init__(self, layers=[2, 2, 1], learning_factor=1, epochs=50,
                 activation_function="sigmoid"):

        self.epochs = epochs
        self.layers = layers
        self.number_of_layers = len(layers)
        self.biases = [2 * np.random.rand(layer, 1) - 1 for layer in
                       layers[1:]]
        self.weights = [2 * np.random.rand(layer, units) - 1 for units, layer
                        in zip(layers[:-1], layers[1:])]
        self.activation_function = th.activation_functions[activation_function]

    def _feedforward(self, sample):
        """Generate the outputs of each layer from a sample

        Args:
            sample: data to feed the network with.

        Returns:
            outputs: output values of each unit in each layer.
            activations: otuput values through the activation function.
        """
        outputs, activations = [], []
        for weight, bias in zip(self.weights, self.biases):
            dot_product = np.dot(weight, sample) + bias
            outputs.append(dot_product)
            activations.append(self.activation_function(dot_product))
        return outputs, activations

    def train(self, training_samples, batch_size):
        """Train the network using stochastic gradient descent (SGD).

        Args:
            training_samples: set of samples to train the network with.
            batch_size: length of each batch for the SGD algorithm.

        Returns:
            No data is returned.
        """
        number_of_samples = len(training_samples)
        for index in xrange(self.epochs):
            random.shuffle(training_samples)
            # Generate the batch of samples in the current iteration.
            batches = [
                training_samples[k:(k + batch_size)]
                for k in xrange(0, number_of_samples, batch_size)
            ]
            print batches
            for batch in batches:
                self._backpropagate_and_update_batch(batch)

    def _backpropagate_and_update_batch(self, batch):

        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        for value, target in batch:
            delta_nabla_bias, delta_nabla_weights = self._backpropagate(value,
                                                                        target)
            nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias,
                                                      delta_nabla_bias)]
            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights,
                                                         delta_nabla_weights)]
        self.biases = [b - (self.learning_factor / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_bias)]
        self.weights = [w - (self.learning_factor / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_weights)]

    def _backpropagate(self, value, target):
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        activations, layers_outputs = [value], []
        layers_outputs, activations = self._feedforward(activations)
