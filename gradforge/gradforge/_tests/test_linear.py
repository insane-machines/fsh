from gradforge.core.tensor import Tensor
from gradforge.optimizers.sgd import SGD
from gradforge.autograd.standard_engine import StandardEngine
import tracemalloc

tracemalloc.start()

class Linear:
    def __init__(self, x: Tensor, y: Tensor):
        self.w = Tensor([], requires_grad=True)
        self.x = x
        self.y = y

    @property
    def params(self) -> list[Tensor]:
        return [attr_val\
                 for attr_val in self.__dict__.values()\
                      if attr_val.requires_grad]
    
    def train(self, epochs):
        self.w.data = self.x.backend.array_zeros((self.x.data.shape[1], y.data.shape[1]))
        engine = StandardEngine()
        optimizer = SGD(learning_rate=1e-3)

        for epoch in range(epochs):
            y_pred  = self.x @ self.w
            error   = self.y - y_pred
            loss    = error ** 2
            engine.backward(loss)
            optimizer.step(self.params)
            optimizer.zero_grad(self.params)

        print(self.x.data @ self.w.data)

x       = Tensor([[4], [8], [9]])
y       = Tensor([[8], [16], [18]])
test    = Tensor([[9], [10]])
assert x.data.shape == y.data.shape

before = tracemalloc.take_snapshot()

model = Linear(x, y)
model.train(epochs=10000)

after = tracemalloc.take_snapshot()

print(test.data @ model.w.data)
print(model.w.data.shape)

stats = after.compare_to(before, "lineno")
for result in stats[:50]:
    print(result)

# 0.757 seconds