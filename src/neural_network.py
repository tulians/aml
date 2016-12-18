# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
#===================================

"""Provides an abstraction to generate neural networks."""

import numpy as np
import threshold as th
import perceptron as p

class NeuralNetwork(object):
    """Layer against layer neural network architecture."""

    def __init__(self, number_of_inputs=2, number_of_outputs=1,
                 hidden_layers=[2], learning_factor=1, epochs=50,
                 activation_function="sigmod"):

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

        self.learning_factor = learning_factor
        self.epochs = epochs

        if activation_function == "unitstep":
            self.activation_function = th.UnitStep()
        else:
            self.activation_function = th.Sigmoid()

        self.layers = self._generate_layers(hidden_layers)
        # TODO: ADD EXCEPTIONS IN CASE ANY PARAMETER IS WRONG.

    def _generate_layers(self, hidden_layers):
        """Generates the hidden layers and the output layer of the network.

        Args:
            hidden_layers: list with the size of each hidden layer.

        Returns:
            layers: hidden layers and output layer of the network.
        """

        weights = []
        # Generate the weights from the input layer to the 1st hidden layer.
        for first_layer_unit in xrange(hidden_layers[0]):
            weights.append(2 * np.random.rand(self.number_of_inputs) - 1)
        # Generate the weights in the hidden layers.
        for hidden_layer in xrange(len(hidden_layers) - 1):
            for hidden_layer_unit in xrange(hidden_layers[hidden_layer]):
                weights.append(2 * np.random.rand(
                    hidden_layers[hidden_layer + 1]) - 1)
        # Generate the weights from the last hidden layer to the output layer.
        for output_layer_unit in xrange(self.number_of_outputs):
            weights.append(2 * np.random.rand(
                hidden_layers[len(hidden_layers) - 1]) - 1)

        layers = []
        # Generate the hidden layers perceptrons
        for layer_id in xrange(len(hidden_layers)):
            start_index = sum(nodes for nodes in hidden_layers[:layer_id])
            end_index = start_index + hidden_layers[layer_id]
            layer = self.NeuronLayer(weights[start_index:end_index],
                                     hidden_layers[layer_id],
                                     self.learning_factor, self.epochs,
                                     self.activation_function)
            layers.append(layer)
        # Generate the output layer perceptron(s).
        output_layer = self.NeuronLayer(weights[-self.number_of_outputs:],
                                        self.number_of_outputs,
                                        self.learning_factor, self.epochs,
                                        self.activation_function)
        layers.append(output_layer)

        return layers

    def _backpropagation(self):
        pass

    def _feedforward(self, input_sample):
        """Computes the output of using 'input_sample' as network input.

        Args:
            input_sample: training sample provided.

        Returns:
            output: output value of the feedforward procedure.
        """
        pass

    def train(self, training_sample, expected_output):
        """Computes the components of the weights vector 'w' across all layers.

        Args:
            training_sample: Sample M-dimensional array.
            expected_output: Array of labels that represent the sample's class.

        Returns:
            w: Weights vector, containing the weights of all the synapses.
        """
        predicted_output = self._feedforward(training_sample)


    class NeuronLayer(object):
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
                layer.append(p.Perceptron(self.weights[unit],
                                          self.learning_factor,
                                          self.epochs,
                                          self.activation_function))
            return layer
