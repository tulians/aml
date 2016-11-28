# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
#===================================

"""Provides an abstraction to generate neural networks."""

import numpy as np
import threshold as th
import perceptron as p

class NeuralNetwork(object):
    """All against all neural rectangular network architecture."""

    def __init__(self, number_of_inputs=2, number_of_outputs=1,
                 number_of_hidden_layers=1, units_per_hidden_layer=2,
                 learning_factor=1, epochs=50, activation_function="sigmod"):

        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

        self.number_of_hidden_layers = number_of_hidden_layers
        self.units_per_hidden_layer = units_per_hidden_layer
        self.learning_factor = learning_factor
        self.epochs = epochs

        if activation_function == "unitstep":
            self.activation_function = th.UnitStep()
        else:
            self.activation_function = th.Sigmoid()

        self.layers = self._generate_layers()

    def _generate_layers(self):
        """Generates the hidden layers and the output layer of the network.

        Args:
            No arguments.

        Returns:
            layers: hidden layers and output layer of the network.
        """

        weights = []
        # Generate the weights from the input layer to the 1st hidden layer.
        for first_layer_unit in xrange(self.units_per_hidden_layer):
            weights.append(2 * np.random.rand(self.number_of_inputs) - 1)
        # Generate the weights in the hidden layers.
        for hidden_layer in xrange(1, self.number_of_hidden_layers):
            for hidden_layer_unit in xrange(self.units_per_hidden_layer):
                weights.append(2 * np.random.rand(self.units_per_hidden_layer) - 1)
        # Generate the weights from the last hidden layer to the output layer.
        for output_layer_unit in xrange(self.number_of_outputs):
            weights.append(2 * np.random.rand(self.units_per_hidden_layer) - 1)

        layers = []
        # Generate the hidden layers perceptrons
        for layer_id in xrange(self.number_of_hidden_layers):
            layer = self.NeuronLayer(weights, self.units_per_hidden_layer,
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
# TODO: ESTO HACELO BIEN. ARMÁ UNA RED QUE SEA DE TAMAÑO ALEATORIO.
        units_outputs = []
        layer_number = 0
        for layer_i in self.layers:
            layer_i_outputs = []
            layer_number += 1
            for unit in layer_i.layer:
                layer_i_outputs.append(unit.output(input_sample))
            units_outputs.append(layer_i_outputs)
            input_sample = units_outputs[layer_number - 1]
        print units_outputs


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
                layer.append(p.Perceptron(self.weights[unit], self.learning_factor,
                                          self.epochs, self.activation_function))
            return layer
