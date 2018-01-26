# DataBuffer data structure module.
# ===================================

try:
    from random import getrandbits
except ImportError:
    from urandom import getrandbits

class DataBuffer(list):
    """Minimal extention of vanilla Python list capabilities."""

# Inheriting from built-in classes is not fully supported.
# Raised https://github.com/micropython/micropython/issues/3554
# as `data` should be sent in __init__.
#    def __init__(self, data=None):
#        super().__init__(data)
#        self.shape = self._shape(data)

    def normalize(self):
        return self._transform(self._methods['nrm'])

    def augment(self):
        return self._transform(self._methods['aug'])

    def mse(self, other):
        if isinstance(other, (list, tuple)):
            return sum((self - other) ** 2)
        raise TypeError('cannot operate with a non-sequence object')
    
    def dot(self, other):
        return sum(i * j for i, j in zip(self, other))

    def randomize(self, rows, columns, bits=8):
        def _randomgenerator(bits):
            return (getrandbits(bits) / (2 ** bits) * 2) - 1
        self[:] = [[_randomgenerator(bits) for c in range(columns)] 
                    for r in range(rows)]

    # Private
    # ESP8266 implementation of Micropython does not have operator.py. 
    def __sub__(self, other):
        return self._operator(self._methods['sub'], other)

    def __add__(self, other):
        return self._operator(self._methods['add'], other)

    def __truediv__(self, other):
        return self._operator(self._methods['div'], other)

    def __mul__(self, other):
        return self._operator(self._methods['mul'], other)

    def __pow__(self, other):
        return self._operator(self._methods['pow'], other)

    def _operator(self, op, other):
        # ESP8266 implementation of Micropython does not have Sequence type.
        if isinstance(other, (list, tuple)):
            if len(other) == len(self):
                return DataBuffer(op(a, b) for a, b in zip(self, other))
            raise ValueError('cannot operate on a sequence of unequal length')
        # In case `other` is a Number. 
        return DataBuffer(op(a, other) for a in self)

    def _shape(self, data):
        if not isinstance(data, (tuple, list)):
            return tuple()
        return (len(data),) + self._shape(data[0])

    def _transform(self, op):
        self.shape = self._shape(self)
        if len(self.shape) == 1:
        # El problema con el append es que retorna null. Hay que tomar
        # ambos casos en cuenta.
            temp = op(self)
            if temp is not None: self[:] = temp
        #    self[:] = op(self)
        else:
            for index, element in enumerate(self): 
                self[index] = DataBuffer(element)._transform(op)
        return self

    _methods = {
        'sub': lambda a, b: a - b,
        'add': lambda a, b: a + b,
        'div': lambda a, b: a / b,
        'mul': lambda a, b: a * b,
        'pow': lambda a, b: a ** b,
        'nrm': lambda d: (d - min(d)) / (max(d) - min(d)),
        'aug': lambda d: d.append(1)
    }
