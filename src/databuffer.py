# DataBuffer data structure module.
# ===================================

from urandom import getrandbits


class DataBuffer(list):
    """Minimal extention of vanilla Python list capabilities."""

    # Algebraic
    def normalize(self):
        min_ = min(self)
        return (self - min_) / (max(self) - min_)

    def augment(self):
        return self.append(1)
    
    def mse(self, other):
        if isinstance(other, (list, tuple)):
            return sum((self - other) ** 2)
        raise TypeError('cannot operate with a non-sequence object')
    
    def dot(self, other):
        return sum(i * j for i, j in zip(self, other))

    def randomize(self, randdims):
        bits, n_elem = 8, 1
        if self is not []: del self[:]
        for n in randdims: n_elem *= n
        for _ in range(randdims[0]):
            self.append(DataBuffer([
                ((getrandbits(bits) / (2 ** bits)) * 2) - 1
                for _ in range(n_elem // randdims[0])
            ]))

    # Private
    # ESP8266 implementation of Micropython does not have operator.py. 
    def __sub__(self, other):
        def _sub(a, b): return a - b
        return self._operator(other, _sub)

    def __add__(self, other):
        def _add(a, b): return a + b
        return self._operator(other, _add)

    def __truediv__(self, other):
        def _td(a, b): return a / b
        return self._operator(other, _td)

    def __mul__(self, other):
        def _mul(a, b): return a * b
        return self._operator(other, _mul)

    def __pow__(self, other):
        def _pow(a, b): return a ** b
        return self._operator(other, _pow)

    def _operator(self, other, op):
        # ESP8266 implementation of Micropython does not have Sequence type.
        if isinstance(other, (list, tuple)):
            if len(other) == len(self):
                return DataBuffer(op(a, b) for a, b in zip(self, other))
            raise ValueError('cannot operate on a sequence of unequal length')
        # In case `other` is a Number. 
        return DataBuffer(op(a, other) for a in self)
