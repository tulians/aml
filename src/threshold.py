# aml - Machine learning library.
# Threshold methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of threshold methods."""

import numpy as np
import math as m


def UnitStep(center=0):
    # return lambda x: 0 if x < center else (0.5 if x == center else 1)
    return lambda x: 0 if x < center else 1


def Sigmoid(center=0, width=1, derivative_needed=False):
    if derivative_needed:
        return (lambda x: m.exp((-x + center) / width) /
                pow((1 + m.exp((-x + center) / width)), 2))
    else:
        return lambda x: 1 / (1 + m.exp((-x + center) / width))


def TanH(center=0, width=1, derivative_needed=False):
    if derivative_needed:
        return lambda x: 1.0 - pow((x - center) / width, 2)
    else:
        return lambda x: np.tanh((x - center) / width)
