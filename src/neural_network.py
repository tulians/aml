# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
# ===================================

"""Provides an abstraction to generate neural networks."""

import utils as u
import numpy as np
import threshold as th
import perceptron as p


class NeuralNetwork(object):
    """Layer against layer neural network architecture."""

    def __init__(self, number_of_inputs=2, number_of_outputs=1, hidden_layers=[2],
                 learning_factor=1, epochs=50, activation_function="sigmod"):

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

        layers, weights = [], []
        # Generate the first hidden layer of perceptrons.
        for first_layer_unit in xrange(hidden_layers[0]):
            weights.append(2 * np.random.rand(self.number_of_inputs + 1) - 1)
        layers.append(self._NeuronLayer(weights,
                                        hidden_layers[0],
                                        self.learning_factor,
                                        self.epochs,
                                        self.activation_function))
        del weights[:]
        # Generate the remaning hidden layers of perceptrons.
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
        # Generate the output layer of perceptrons.
        for output_layer_unit in xrange(self.number_of_outputs):
            weights.append(2 * np.random.rand(hidden_layers[-1] + 1) - 1)
        layers.append(self._NeuronLayer(weights,
                                        self.number_of_outputs,
                                        self.learning_factor,
                                        self.epochs,
                                        self.activation_function))
        del weights[:]

        return layers

    def _error_output_layer(self, expected_output, layers_output):
        """Makes necessary corrections to output's layer weights

        Args:
            expected_output: class label value to match.
            layers_output: feedforward output values from every layer.

        Returns:
            output_layer_weights_fixes: corrections to output layer's weights.
        """

        predicted_output = layers_output[-1]
        output_layer_inputs = layers_output[-2]
        output_layer_inputs.append(1)

        error_output_layer = np.array([0.0] * self.number_of_outputs)

        for i in xrange(self.number_of_outputs):
            error = expected_output[i] - predicted_output[i]
            # TODO: Check to use the derivative of the activation_function.
            error_output_layer[i] = (predicted_output[i] *
                                     (1 - predicted_output[i]) * error)

        number_of_weights = len(output_layer_inputs) * self.number_of_outputs
        print len(output_layer_inputs), self.number_of_outputs, number_of_weights
        output_layer_weights_fixes = np.array([0.0] * number_of_weights)
        for weight_index in xrange(number_of_weights):
            curr_output = weight_index / len(output_layer_inputs)
            curr_input = weight_index / self.number_of_outputs
            print curr_output, curr_input
            output_layer_weights_fixes[weight_index] = (error_output_layer[curr_output] *
                                                        output_layer_inputs[curr_input])

        return output_layer_weights_fixes


    # TODO: COMPLETE!
    def _error_hidden_layers(self, expected_output, layers_output, training_sample):
        pass

    def _update_weights(self, output_layer_error, hidden_layers_error):
        pass

    def _backpropagation(self, expected_output, layers_output, training_sample):
        """Adjusts network's weights by using backpropagation of errors.

        Args:
            expected_output: class label value to match.
            layers_output: feedforward output values from every layer.
            training_sample: inputs to the network.

        Returns:
            No return values.
        """
        # Calculate the compensation for the output layer's weights.
        output_layer_error = self._error_output_layer(expected_output, layers_output)
        # Calculate the compensation for the hidden layers' weights.
        hidden_layers_error = self._error_hidden_layers(expected_output, layers_output,
                                                       training_sample)
        self._update_weights(output_layer_error, hidden_layers_error)

    def _feedforward(self, sample):
        """Computes the output of using 'sample' as network's input.

        Args:
            sample: training sample provided.

        Returns:
            outputs: output value of the feedforward procedure.
        """
        outputs = []
        for hidden_layer in self.layers:
            outputs.append([layer.output(sample) for layer in hidden_layer.layer])
            sample = outputs[-1]
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
        layers_output = self._feedforward(training_sample)
        sse = u.sum_of_squared_errors(layers_output[-1], expected_output)
        self._backpropagation(expected_output, layers_output, training_sample)
        print(str(layers_output[-1]) + " " + str(expected_output))

    def run(self, input_sample, expected_output, epochs=1000000):
        for _ in xrange(epochs):
            self.train(input_sample, expected_output)

    class _NeuronLayer(object):
        """Inner class that generates a layer of perceptrons."""

        def __init__(self, weights, units_per_layer, learning_factor, epochs,
                     activation_function):

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
