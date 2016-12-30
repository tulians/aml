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
            compensations: corrections to output layer's weights.
        """

        predicted_output = layers_output[-1]
        output_layer_inputs = layers_output[-2]
        output_layer_inputs.append(1)

        partial_error = np.array([0.0] * self.number_of_outputs)
        # General terms for all weights.
        for i in xrange(self.number_of_outputs):
            error = self._dEtotal_wrt_dOutput(i, expected_output,
                                              predicted_output)
        #    error = expected_output[i] - predicted_output[i]
            partial_error[i] = (predicted_output[i] *
                                (1 - predicted_output[i]) * error)
        # Weight specific corrections.
        number_of_weights = (len(output_layer_inputs) *
                             self.number_of_outputs)
        compensations = np.array([0.0] * number_of_weights)
        for weight_index in xrange(number_of_weights):
            curr_output = weight_index / len(output_layer_inputs)
            curr_input = weight_index / self.number_of_outputs
            compensations[weight_index] = (partial_error[curr_output] *
                                           output_layer_inputs[curr_input]
                                           )

        return compensations

###############################################################################
    def _error_hidden_layers(self, expected_output, layers_output,
                             layers_input, training_sample):
        hidden_layers = self.layers[:-1][0].layer
        # hidden_layers.reverse()
        layers_input.append(1)

        compensations = []

        number_of_weights = len(hidden_layers) * len(layers_input)

        for weight in xrange(number_of_weights):
            sum_fst_pd = 0
            for o in xrange(self.number_of_outputs):
                sum_fst_pd += (self._dEtotal_wrt_dOutput(o, expected_output,
                                                         layers_output[-1]) *
                               self._dOutput_wrt_dInput(layers_output, 0,
                                                        weight) *
                               self._dInput_wrt_dW(layers_input, 0, weight))
            second_pd = self._dOutput_wrt_dInput()
            third_pd = self._dInput_wrt_dW()
    # end

    def _dEtotal_wrt_dOutput(self, output_num, expected_output,
                             predicted_output):
        return predicted_output[output_num] - expected_output[output_num]

    def _dOutput_wrt_dInput(self, layers_output, layer_num, perceptron_num):
        output = layers_output[layer_num].layer[perceptron_num]
        return output * (1 - output)

    def _dInput_wrt_dW(self, layers_input, layer_num, weight_num):
        return layers_input[layer_num][weight_num]


###############################################################################

    def _update_weights(self, output_layer_error, hidden_layers_error):
        pass

    def _backpropagation(self, expected_output, layers_output, layers_input,
                         training_sample):
        """Adjusts network's weights by using backpropagation of errors.

        Args:
            expected_output: class label value to match.
            layers_output: feedforward output values from every layer.
            training_sample: inputs to the network.

        Returns:
            No return values.
        """
        # Calculate the compensation for the output layer's weights.
        output_layer_error = self._error_output_layer(expected_output,
                                                      layers_output)
        # Calculate the compensation for the hidden layers' weights.
        hidden_layers_error = self._error_hidden_layers(expected_output,
                                                        layers_output,
                                                        layers_input,
                                                        training_sample)
        self._update_weights(output_layer_error, hidden_layers_error)

    def _feedforward(self, sample):
        """Computes the output of using 'sample' as network's input.

        Args:
            sample: training sample provided.

        Returns:
            outputs: output value of the feedforward procedure.
            inputs: input value of each layer.
        """
        outputs, inputs = [], []
        for hidden_layer in self.layers:
            inputs.append(sample)
            outputs.append([layer.output(sample) for layer in
                            hidden_layer.layer])
            sample = outputs[-1]
        return outputs, inputs

    def train(self, training_sample, expected_output):
        """Computes the components of the weights vector 'w' across all layers.

        Args:
            training_sample: Sample M-dimensional array.
            expected_output: Array of labels that represent the sample's class.

        Returns:
            w: Weights vector, containing the weights of all the synapses.
        """

        # TODO: Add TypeError management.
        layers_output, layers_input = self._feedforward(training_sample)
        sse = u.sum_of_squared_errors(layers_output[-1], expected_output)
        self._backpropagation(expected_output, layers_output, layers_input,
                              training_sample)
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
