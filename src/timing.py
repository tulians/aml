# aml - Machine learning library.
# Timing methods module.
# Author: Julian Ailan
#===================================

"""Provides a set of custom timing methods."""

import time

def TimeMethod(method, values):
    """Measures the time in seconds of a specific method."""
    start = time.clock()
    return_value = method(values)
    return (return_value, time.clock() - start)
