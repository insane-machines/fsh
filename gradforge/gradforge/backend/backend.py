from abc import ABC, abstractmethod
class Backend(ABC):

    @abstractmethod
    def convert(data):
        pass

    #Binary operations
    @abstractmethod
    def add(a, b):
        pass

    @abstractmethod
    def div(a, b):
        pass

    @abstractmethod
    def mul(a, b):
        pass

    @abstractmethod
    def sub(a, b):
        pass

    #Matrix operations
    @abstractmethod
    def matmul(a, b):
        pass

    @abstractmethod
    def transpose(matrix):
        pass

    #Unary operations
    @abstractmethod
    def relu(x):
        pass

    @abstractmethod
    def sigmoid(x):
        pass

    @abstractmethod
    def tanh(x):
        pass

    @abstractmethod
    def softmax(x):
        pass

    
