from .engine import Engine
from ..core.tensor import Tensor

class StandardEngine(Engine):
    def __init__(self) -> None:
        super().__init__()

    def _prepare_root_grad(self, tensor: Tensor):
        return self._backend.ones_like(tensor.data)

    def _tensor_backward(self, tensor: Tensor, grad) -> list:
        if tensor.grad_fn is None:
            return []

        return tensor.grad_fn.backward(grad)
        
    def _start_backward_recursion(self, tensor_backward):
        for parent, parent_grad in tensor_backward:
            if parent.grad_fn is not None:  
                self.backward(parent, parent_grad)

    def backward(self, tensor: Tensor, grad=None):
        if grad is None:
            grad = self._prepare_root_grad(tensor)

        tensor_backward = self._tensor_backward(tensor, grad)
        self._start_backward_recursion(tensor_backward)