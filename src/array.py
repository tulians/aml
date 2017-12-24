# Array data structure module.
# ===================================

class Array(list):
    """Minimal extention of vanilla Python lists capabilities."""

    def normalize(self):
        min_ = min(self)
        return (self - min_) / (max(self) - min_)

    def augment(self):
        return self.append(1)
    
    def mse(self, other):
        if isinstance(other, (list, tuple)):
            return sum((self - other) ** 2)

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
                return Array(op(a, b) for a, b in zip(self, other))
            raise ValueError('cannot operate on a sequence of unequal length')
        return Array(op(a, other) for a in self)
