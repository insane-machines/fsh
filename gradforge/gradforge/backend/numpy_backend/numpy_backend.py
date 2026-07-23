from ..backend import Backend
import numpy as np

class NumpyBackend(Backend):

    def convert(self, data) -> np.ndarray:
        return np.array(data, dtype=np.float32)

    #Binary operations
    def add(self, a, b):
        return np.add(a, b)

    def div(self, a, b):
        return np.divide(a, b)

    def mul(self, a, b):
        return np.multiply(a, b)

    def sub(self, a, b):
        return np.subtract(a, b)

    #Matrix operations
    def matmul(self, a, b):
        return np.matmul(a, b)

    def transpose(self, matrix):
        return matrix.T

    #Unary operations
    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        pass

    def tanh(self, x):
        pass

    def softmax(self, x):
        pass

    def mean(self, array):
        return np.mean(array)

    def std(self, array):
        return np.std(array)

    def abs(self, array):
        return np.abs(array)

    def zeros_like(self, array):
        return np.zeros_like(array)

    def ones_like(self, array):
        return np.ones_like(array)
    
