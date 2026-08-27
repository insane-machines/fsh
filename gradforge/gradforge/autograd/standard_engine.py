from .engine import Engine
from ..core.tensor import Tensor

class StandardEngine(Engine):
    def __init__(self) -> None:
        super().__init__()

    def _prepare_root_grad(self, tensor: Tensor):
        root_grad = self._backend.ones_like(tensor.data)
        return root_grad

    def _tensor_backward(self, tensor: Tensor, grad):
        if tensor.grad_fn is None:
            return ()
        
        parents = tensor.grad_fn[0]
        grads = tensor.grad_fn[1].backward(grad)


        return zip(parents, grads)
        
    def _start_backward_recursion(self, tensor_backward):
        for parent, parent_grad in tensor_backward:
            if parent.requires_grad:
                parent.grad += parent_grad

            if parent.grad_fn is not None:  
                self.backward(parent, parent.grad)

    def backward(self, tensor: Tensor, grad=None):
        if grad is None:
            grad = self._prepare_root_grad(tensor)

        tensor_backward = self._tensor_backward(tensor, grad)
        self._start_backward_recursion(tensor_backward)