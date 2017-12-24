# Threshold methods module.
# ===================================

"""Provides a set of threshold functions and its derivatives."""

import math as m

EPSILON = 0.1

class Threshold(object):
    def __init__(self, th_fun):
        self.th = activations[th_fun]
        self.th_prime = derivatives[th_fun]

    def __call__(self, data, is_derivative=False):
        if not is_derivative:
            return [self.th(elem) for elem in data]
        return [self.th_prime(elem) for elem in data]

  
activations = {
    'sin': lambda x: m.sin(x),
    'cos': lambda x: m.cos(x),
    'tanh': lambda x: m.tanh(x),
    'logistic': lambda x: 2 * (1.0 / (1.0 + m.exp(-x))) - 1,
    'relu': lambda x: max(x * EPSILON, x)
}


derivatives = {
    'sin': lambda x: m.cos(x),
    'cos': lambda x: -m.sin(x),
    'tanh': lambda x: 1.0 - (m.tanh(x) ** 2),
    'logistic': lambda x: 2 * (1.0 / (1.0 + m.exp(-x))) - 1, 
    'relu': lambda x: 1 if x > 0 else EPSILON
}
