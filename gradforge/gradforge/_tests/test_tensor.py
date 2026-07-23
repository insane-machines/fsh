from gradforge.core.tensor import Tensor

class Linear:
    def train(self, x, y):
        w = Tensor(0, requires_grad=True)
        for i in range(100):
            y_pred =  x * w
            loss = y - y_pred
            if loss.grad_fn is not None:
                loss.backward()
            w.data -= w.grad * loss.data * 0.001
            print(w.data)

        assert w.data == 2

x = Tensor(4)
y = Tensor(8)
model = Linear()
model.train(x, y)


