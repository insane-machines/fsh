from abc import ABC, abstractmethod
from ..backend.backend_manager import BackendManager

class Engine(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._backend = BackendManager.get_backend()

    @abstractmethod
    def _prepare_root_grad():
        pass

    @abstractmethod
    def _tensor_backward():
        pass
    
    @abstractmethod
    def _start_backward_recursion():
        pass

    @abstractmethod
    def backward():
        pass