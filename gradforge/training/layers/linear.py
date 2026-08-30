from .layer import Layer
from ...core.tensor import Tensor
from ...backend.backend_manager import BackendManager

class Linear(Layer):
    def __init__(self, x: Tensor, out_shape) -> None:
        super().__init__()      
        self.out_shape = out_shape
        self._backend = BackendManager.get_backend()
        self.x = x
        self.w: Tensor = Tensor(self._backend.array_ones((x.data.shape()[1], out_shape)), requires_grad=True)

    def predict(self) -> Tensor:
        return self.x @ self.w

    @property
    def params(self) -> list[Tensor]:
        return [attr_val\
                    for attr_val in self.__dict__.values()\
                        if attr_val.requires_grad]