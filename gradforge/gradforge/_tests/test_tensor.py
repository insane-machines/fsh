from gradforge.core.tensor import Tensor
from ..optimizers.sgd import SGD
from ..autograd.standard_engine import StandardEngine
class Linear:
    def train(self, x: Tensor, y: Tensor):
        w = Tensor(x.backend.array_zeros((x.data.shape[1], y.data.shape[1])), requires_grad=True)
        print(w.data.shape)
        engine = StandardEngine()
        optimizer = SGD(learning_rate=1e-3)
        for i in range(200):
            y_pred =  x @ w 
            error = y - y_pred
            loss = error ** 2
            engine.backward(loss)
            optimizer.step([w])
            optimizer.zero_grad([w])
            print(f"Weight: {w.data}")

        res = x.data @ w.data
        assert res.data.shape == y.data.shape

x = Tensor([[4, 8]])
y = Tensor([[8, 16]])
model = Linear()
print(x.data.shape, y.data.shape)
model.train(x, y)


