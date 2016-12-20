# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
# ===================================

"""Provides an abstraction to generate neural networks."""

import numpy as np
import threshold as th
import perceptron as p


class NeuralNetwork(object):
    """Layer against layer neural network architecture."""

    def __init__(self, number_of_inputs=2, number_of_outputs=1,
                 hidden_layers=[2], learning_factor=1, epochs=50,
                 activation_function="sigmod"):
        # TODO: ADD EXCEPTIONS IN CASE ANY PARAMETER IS WRONG.
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

        self.learning_factor = learning_factor
        self.epochs = epochs

        if activation_function == "unitstep":
            self.activation_function = th.UnitStep()
        else:
            self.activation_function = th.Sigmoid()

        self.layers = self._generate_layers(hidden_layers)

    def _generate_layers(self, hidden_layers):
        """Generates the hidden layers and the output layer of the network.

        Args:
            hidden_layers: list with the size of each hidden layer.

        Returns:
            layers: hidden layers and output layer of the network.
        """

        layers, weights = list(), list()
        # Generate the first layer of perceptrons..
        for first_layer_unit in xrange(hidden_layers[0]):
            weights.append(2 * np.random.rand(self.number_of_inputs + 1) - 1)
        layers.append(self._NeuronLayer(weights,
                                        hidden_layers[0],
                                        self.learning_factor,
                                        self.epochs,
                                        self.activation_function))
        del weights[:]

        for hidden_layer in xrange(1, len(hidden_layers)):
            for hidden_layer_unit in xrange(hidden_layers[hidden_layer]):
                weights.append(2 * np.random.rand(hidden_layers[hidden_layer -
                                                                1] + 1) - 1)
            layers.append(self._NeuronLayer(weights,
                                            hidden_layers[hidden_layer],
                                            self.learning_factor,
                                            self.epochs,
                                            self.activation_function))
            del weights[:]

        for output_layer_unit in xrange(self.number_of_outputs):
            weights.append(2 * np.random.rand(
                hidden_layers[len(hidden_layers) - 1] + 1) - 1)
        layers.append(self._NeuronLayer(weights,
                                        self.number_of_outputs,
                                        self.learning_factor,
                                        self.epochs,
                                        self.activation_function))
        del weights[:]

        return layers

    def _backpropagation(self, predicted_output, expected_output):
        pass

    def _feedforward(self, sample):
        """Computes the output of using 'sample' as network input.

        Args:
            sample: training sample provided.

        Returns:
            output: output value of the feedforward procedure.
        """
        for hidden_layer in self.layers:
            outputs = []
            for hidden_layer_unit in hidden_layer.layer:
                outputs.append(hidden_layer_unit.output(sample))
            sample = outputs
            print "Out: " + str(outputs)
        return outputs

    def train(self, training_sample, expected_output):
        """Computes the components of the weights vector 'w' across all layers.

        Args:
            training_sample: Sample M-dimensional array.
            expected_output: Array of labels that represent the sample's class.

        Returns:
            w: Weights vector, containing the weights of all the synapses.
        """
        # TODO: Add TypeError management.
        predicted_output = self._feedforward(training_sample)
        print predicted_output
        self._backpropagation(predicted_output, expected_output)

    class _NeuronLayer(object):
        """Inner class that generates a layer of perceptrons."""

        def __init__(self, weights, units_per_layer, learning_factor,
                     epochs, activation_function):

            self.weights = weights
            self.units_per_layer = units_per_layer
            self.learning_factor = learning_factor
            self.epochs = epochs
            self.activation_function = activation_function

            self.layer = self._generate_layer()

        def _generate_layer(self):
            """Generates a layer of perceptrons.

            Args:
                No arguments.

            Returns:
                layer: column of perceptrons with the given characteristics.
            """

            layer = []
            for unit in xrange(self.units_per_layer):
                print "unit #" + str(unit) + " " + str(self.weights[unit])
                layer.append(p.Perceptron(self.weights[unit],
                                          self.learning_factor,
                                          self.epochs,
                                          self.activation_function))
            return layer
