# aml - Machine learning library.
# Timing methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of custom timing methods."""

# Built-in modules.
import time


def _TimeMethod(method, values):
    """Measures the time in seconds of a specific method."""
    start = time.clock()
    return_value = method(values)
    return (return_value, time.clock() - start)


def TimeElapsed(statements, iterations=10):
    """Average execution time of snippets of code."""
    time_elapsed = 0
    for _ in xrange(iterations):
        start = time.clock()
        exec(statements)
        time_elapsed += time.clock() - start
    print("Elapsed time after {0} iterations: {0}".format(iterations,
                                                          time_elapsed /
                                                          iterations))
