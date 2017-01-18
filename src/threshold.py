# aml - Machine learning library.
# Threshold methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of threshold functions."""

# Third-party modules
import numpy as np


def logistic(data):
    output = 1.0 / (1.0 + np.exp(-data))
    return (2 * output) - 1


def logistic_prime(data):
    return 0.5 * ((1 / np.cosh(data / 2.0)) ** 2)


def tan_h(data):
    return np.tanh(data)


def tan_h_prime(data):
    return 1.0 - np.tanh(data) ** 2


def unit_step(data, center=0):
    return 0 if data < center else 1


def relu(data, epsilon=0.1):
    return np.maximum(epsilon * data, data)


def relu_prime(data, epsilon=0.1):
    gradients = 1. * (data > 0)
    gradients[gradients == 0] = epsilon
    return gradients


def softsign(data):
    return data / (1 + np.abs(data))


def softsign_prime(data):
    return 1 / ((1 + np.abs(data)) ** 2)


def arctan(data):
    return np.arctan(data)


def arctan_prime(data):
    return 1 / (1 + (data ** 2))


def softplus(data):
    return np.log(1 + np.exp(data))


def softplus_prime(data):
    return 1 / (1 + np.exp(-data))


def bent_identity(data):
    return ((np.sqrt((data ** 2) + 1) - 1) / 2) + data


def bent_identity_prime(data):
    return (data / (2 * np.sqrt((data ** 2) + 1))) + 1


def sin_funct(data):
    return np.sin(data)


def sin_prime(data):
    return np.cos(data)


def identity(data):
    return data


def identity_prime(data):
    return 1


activation_functions = {
    "logistic": logistic,
    "tanh": tan_h,
    "unitstep": unit_step,
    "relu": relu,
    "softsign": softsign,
    "arctan": arctan,
    "softplus": softplus,
    "bentidentity": bent_identity,
    "sin": sin_funct,
    "identity": identity
}

activation_derivatives = {
    "logistic": logistic_prime,
    "tanh": tan_h_prime,
    "relu": relu_prime,
    "softsign": softsign_prime,
    "arctan": arctan_prime,
    "softplus": softplus_prime,
    "bentidentity": bent_identity_prime,
    "sin": sin_prime,
    "identity": identity_prime,
}
