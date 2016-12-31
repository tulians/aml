# aml - Machine learning library.
# Testing module.
# Author: Julian Ailan
# ===================================

# Built-in modules.
import unittest


class PerceptronTest(unittest.TestCase):

    def test_training_set(self):
        import datasets as ds
        import perceptron as p
        import numpy as np
        N = 100
        set1 = ds.generate_multivariate_normal_set(N, [-1.5, -1.5])
        set2 = ds.generate_multivariate_normal_set(N, [1.5, 1.5])
        dataset = np.concatenate((set1, set2), axis=0)
        labels = np.concatenate((np.ones(N, dtype=int),
                                 (-1) * np.ones(N, dtype=int)), axis=0)
        per = p.Perceptron()
        w = per.train(dataset, labels)

        correct_set_1, correct_set_2 = 0, 0

        for i in xrange(N):
            if np.dot(w, np.concatenate((dataset[i], [1]), axis=0)) > 0:
                correct_set_1 += 1
        for i in xrange(N, 2*N):
            if np.dot(w, np.concatenate((dataset[i], [1]), axis=0)) < 0:
                correct_set_2 += 1

        print correct_set_1, correct_set_2

        self.assertTrue(((correct_set_1 > 0.95 * N) and
                         (correct_set_2 > 0.95 * N)))

if __name__ == '__main__':
    unittest.main()
