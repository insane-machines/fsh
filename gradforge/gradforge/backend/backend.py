from abc import ABC, abstractmethod
class Backend(ABC):

    @abstractmethod
    def convert(self, data):
        return data

    #Binary operations
    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def div(self, a, b):
        pass

    @abstractmethod
    def mul(self, a, b):
        pass

    @abstractmethod
    def sub(self, a, b):
        pass

    #Matrix operations
    @abstractmethod
    def matmul(self, a, b):
        pass

    @abstractmethod
    def transpose(self, matrix):
        return matrix

    #Unary operations
    @abstractmethod
    def relu(self, x):
        return x

    @abstractmethod
    def sigmoid(self, x):
        return x

    @abstractmethod
    def tanh(self, x):
        return x

    @abstractmethod
    def softmax(self, x):
        return x

    @abstractmethod
    def mean(self, array):
        return array

    @abstractmethod
    def std(self, array):
        return array

    @abstractmethod
    def abs(self, array):
        return array

    @abstractmethod
    def zeros_like(self, array):
        return array

    @abstractmethod
    def ones_like(self, array):
        return array

    @abstractmethod
    def array_zeros(self, shape):
        return shape

    @abstractmethod
    def array_ones(self, shape):
        return shape

    @abstractmethod
    def is_native(self, data) -> bool:
        pass



    
