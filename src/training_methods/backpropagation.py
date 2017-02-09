# aml - Machine learning library.
# Backpropagation alternatives module.
# Author: Julian Ailan
# ===================================

# Third-party modules
import numpy as np


def SGD(layers, weights, activation_function, activation_derivative,
        training_samples, labels, feedforward, cost, learning_rate,
        epochs, ret_error, tolerance, display):
    """Trains the network using stochastic gradient descent (SGD).
    Args:
        training_samples: list of samples used to train the network's
        weights.
        labels: outputs associated to the training_samples.
        learning_rate: 'speed' at which the SGD algorithm learns.
        epochs: number of iterations to perform in SGD.
    Returns:
        If required, a list of squared sum of errors along epochs is
        returned.
    """
    training_error = []
    labels_dims_matches = all([len(x) == layers[-1] for x in labels])

    if not labels_dims_matches:
        print("The entered labels do not have the same dimensions as the"
              " network output layer. These labels are {0}-dimensional"
              " while the output layer is {1}-dimensional.".format(
                  labels.ndim, layers[-1]
              ))
        return
    for epoch in xrange(epochs):
        sample_index = np.random.randint(training_samples.shape[0])
        activations = [training_samples[sample_index]]
        # Forward pass.
        for weight in xrange(len(weights)):
            activations.append(
                activation_function(
                    np.dot(activations[weight], weights[weight])))
        # Backpropagation starts:
        # 1- Output layer weights compensation.
        dEtotal_wrt_dOutput = labels[sample_index] - activations[-1]
        dOutput_wrt_dInput = activation_derivative(activations[-1])
        deltas = [dEtotal_wrt_dOutput * dOutput_wrt_dInput]
        # 2- Hidden layers weights compensation.
        for layer in xrange(len(activations) - 2, 0, -1):
            deltas.append(
                deltas[-1].dot(weights[layer].T) *
                activation_derivative(activations[layer]))
        deltas.reverse()
        # 3- Weights update.
        for index in xrange(len(weights)):
            layer = np.atleast_2d(activations[index])
            delta = np.atleast_2d(deltas[index])
            weights[index] += learning_rate * np.dot(layer.T, delta)

        if ret_error:
            training_error.append(
                cost(
                    feedforward(training_samples),
                    labels
                ))
            if len(training_error) > 1:
                if (abs(training_error[-1] - training_error[-2]) <
                        tolerance):
                    print("Exiting in epoch {0}.".format(epoch))
                    break

    if ret_error:
        if display:
            display(range(len(training_error)), training_error)
        return training_error


methods = {
    "SGD": SGD
}
