# Spark
Spark a general purpose machine learning library aimed a code legibility, great execution speed and minimal dependencies. The only package needed is **NumPy** , which was chosen for its powerful linear algebra and N-dimensional array properties.

### Example
Imagine you need to learn a function like AND, OR, NOT, or any linearly separable boolean function, Spark provides a **Perceptron** class that is able to generate a separation surface among both classes. Taking the AND function as an example, the weights of the separation surface can be coded computed with these few lines:


```
#!python
>>> import perceptron as p
>>> per = p.Perceptron(activation_function = "unitstep")
>>>
>>> data = [[0,0],[0,1],[1,0],[1,1]]
>>> labels = [0,0,0,1]
>>>
>>> per.train(data, labels)
>>>
>>> # Test the classifier
>>> per.output([0,0])
0
>>> per.output([0,1])
0
>>> per.output([1,0])
0
>>> per.output([1,1])
1
```

### Installation


### Todos
  - Add License.
  - Add Instalation steps.

### Version
0.1

### License
