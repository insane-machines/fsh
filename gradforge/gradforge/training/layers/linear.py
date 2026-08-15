from .layer import Layer
from ...core.tensor import Tensor
from ...backend.backend_manager import BackendManager

class Linear(Layer):
    def __init__(self, x: Tensor, out_shape) -> None:
        super().__init__()      
        self.out_shape = out_shape
        self._backend = BackendManager.get_backend()
        self.w: Tensor = Tensor(self._backend.array_ones((x.data.shape()[1], out_shape)), requires_grad=True)

    def train(self):
        pass

    def return_parameters(self):
        pass