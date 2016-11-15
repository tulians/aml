# aml - Machine learning library.
# Neural network module.
# Author: Julian Ailan
#===================================

"""Provides an abstraction to generate neural networks."""

import numpy as np
import threshold as th
import perceptron as p

class NeuralNetwork(object):
    """All against all neural network architecture."""

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

        self.synapses_weights = self.init_weights(number_of_inputs,
                                                  number_of_outputs,
                                                  number_of_hidden_layers,
                                                  units_per_hidden_layer)
# TODO: DONT GENERATE ALL AT ONCE, MAKE LAYERS AND CONNECT LAYERS!

# TODO: Generate an array of different perceptron objects more efficiently.
# TODO: Set the weights vector in each new instance. ITS VARIABLE! Initialize
#       weights and perceptrons in the same method!
        total_number_of_units = ((number_of_hidden_layers *
                                  units_per_hidden_layer) + number_of_outputs)
        self.hidden_layers_units = []
        for unit in xrange(total_number_of_units):
            self.hidden_layers_units.append(p.Perceptron())

        print len(self.synapses_weights), len(self.hidden_layers_units)

    def init_weights(self, number_of_inputs, number_of_outputs,
                     number_of_hidden_layers, units_per_hidden_layer):
        """Initializes the weights of the network synapses.

        Args:
            number_of_inputs: Number of input weights to the first hidden layer.
            number_of_outputs: Number of output weights from the last hidden
                               layer.
            number_of_hidden_layers: Number of layers of hidden units.
            units_per_hidden_layer: Number of processing units per hidden layer.

        Returns:
            w: Initial neural network synapses weights.
        """

        number_of_hidden_weights = (pow(units_per_hidden_layer, 2) *
                                    (number_of_hidden_layers - 1))

        number_of_input_weights = number_of_inputs * units_per_hidden_layer

        number_of_output_weights = number_of_outputs * units_per_hidden_layer

        total_number_of_weights = (number_of_input_weights +
                                   number_of_hidden_weights +
                                   number_of_output_weights)

        return 2 * np.random.rand(total_number_of_weights) - 1

    def feedforward(self, input_sample):
        """Computes the output of using 'input_sample' as network input.

        Args:
            input_sample: training sample provided.

        Returns:
            output: output value of the feedforward procedure.
        """
        output = []
# TODO: This section must be completed only when Perceptrons are initialized
#       with their respective weights.
        # Perform the dot product of the inputs with the first hidden layer
        #first_layer_units = self.hidden_layers_nodes[:units_per_hidden_layer]
        #for first_hidden_layer_unit in first_hidden_layer_units:
        #    output.append(first_hidden_layer_unit.output(input_sample))

# TODO: Erase this section when done.
        # Perform the dot products of the inputs & outputs of hidden layers
        #start = units_per_hidden_layer
        #end = (units_per_hidden_layer - 1) * number_of_hidden_layers
        #for hidden_unit in self.hidden_layers_nodes[start:end]:
        #    pass
        # Perform the dot product of the last layer outputs with the output layer

    def train(self, training_sample, expected_output):
        """Computes the components of the weights vector 'w' across all layers.

        Args:
            training_sample: Sample M-dimensional array.
            expected_output: Array of labels that represent the sample's class.

        Returns:
            w: Weights vector, containing the weights of all the synapses.
        """
        predicted_output = self.feedforward(training_sample)
