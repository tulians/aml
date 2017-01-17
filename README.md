# aml
*aml* (accelerated machine learning) is a general purpose machine learning library aimed a code legibility, great execution speed and minimal dependencies. One of the  needed packages is **NumPy**, which was chosen for its powerful linear algebra and N-dimensional array properties. This library is currently under development, so any errors or non-optimal implementations will be fixed in further commits.

### Examples
#### Fully connected neural network
*aml* provides a simple fully connected neural network which uses the classical backpropagation algorithm for learning. The paradigmatic XOR problem can be solved using these lines of code:

```python
>>> import neural_network as nn
>>> net = nn.NeuralNetwork(layers=[2, 2, 1], activation_function="tanh")
>>> training_samples = [
...    [0, 0], [0, 1], [1, 0], [1, 1]
... ]
>>> labels = [0, 1, 1, 0]
>>> net.train(training_samples, labels)
>>> # Test the network.
>>> net.predict(training_samples)
([0, 0], array([[ 0.00010954]]))
([0, 1], array([[ 0.99643299]]))
([1, 0], array([[ 0.99648988]]))
([1, 1], array([[ 1.81847912e-05]]))
```

#### Perceptron
Imagine you need to learn a function like AND, OR, NOT, or any linearly separable boolean function, *aml* provides a **Perceptron** that is able to generate a separation surface among both classes. Taking the AND function as an example, the weights of the separation surface can be coded computed with these few lines:

```python
>>> import perceptron as p
>>> per = p.Perceptron(activation_function = "unitstep")
>>> training_samples = [[0, 0],[0, 1],[1, 0],[1, 1]]
>>> labels = [0, 0, 0, 1]
>>> per.train(training_samples, labels)
>>> # Test the classifier.
>>> for sample in training_samples:
...   print(sample, per.output(sample))
...
([0, 0], 0)
([0, 1], 0)
([1, 0], 0)
([1, 1], 1)
```

### Installation
The only major dependency needed for the library to work is **NumPy**. This package is generally already installed in many major Linux distros. In Mac OS X, it can be installed using **pip**:
```sh
$ pip install numpy
```

### Todos
  - Bug in FNN training outputs, if the NN has 2 outputs and only 1-dim labels are provided, it still trains it.

### Version
0.2

### License
MIT
