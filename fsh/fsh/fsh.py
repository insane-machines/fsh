from math import nan
import numpy as np

class Linear():
    def __init__(self, lr = 0.001):
        self.lr = lr
        self.weight = 1
        self.bias = 1
    def predict(self, x, training=False):
        y_pred = self.weight*x + self.bias
        if not training:
            print(f'▶ Predicted output for x = {x}: {y_pred}')
        return y_pred
    def mse(self, y, y_pred, training=False):
        loss = np.mean((y-y_pred)**2)
        if not training:
            print(f'MSE: {loss}')
        return loss
    def calc_diff(self, param_name, x, y, d_h=1e-5, training=True):
        orig_num = getattr(self, param_name)
        
        setattr(self, param_name, orig_num+d_h)
        loss_p = self.mse(y, self.predict(x, training), training)
        
        setattr(self, param_name, orig_num-d_h)
        loss_m = self.mse(y, self.predict(x, training), training)
        
        setattr(self, param_name, orig_num)

        grad = (loss_p - loss_m) / (2 * d_h)
        return grad
        
    def train(self, x, y, epochs=50, val_data=None, training=True, view_epoch=1, batch_size=1, \
                                                                                    diff_height=1e-5):
        if type(val_data) == list:
            if len(val_data) < 2:
                raise DataError(f'validation list is out of range, expected: 2, got: {len(val_data)}')
            val_x = val_data[0]
            val_y = val_data[1]
            if type(val_x) != np.ndarray or type(val_y) != np.ndarray:
                raise DataError(f'InvalidType: needs <np.ndarray()>, got {type(val_data)}')
        elif val_data == None:
            val_x = None
            val_y = None
        else:
            raise DataError('val_data is not a list')
        for epoch in range(1, epochs+1):
            if type(x) != np.ndarray or type(y) != np.ndarray:
                raise DataError(f'TypeError: Invalid type of training data: needs <np.ndarray>')

            if len(x) > len(y) or len(y) > len(x):
                raise MatchError('Sizes of datas is not match')

            x_data = np.split(x, np.arange(batch_size, len(x), batch_size))
            y_data = np.split(y, np.arange(batch_size, len(y), batch_size))

            for batch in range(len(x_data)):
                x_batch = x_data[batch]
                y_batch = y_data[batch]
                grad_w = self.calc_diff('weight', x_batch, y_batch, d_h=diff_height)
                grad_b = self.calc_diff('bias', x_batch, y_batch, d_h=diff_height)
            
                self.weight -= self.lr * grad_w
                self.bias -= self.lr * grad_b

                y_pred = self.predict(x_batch, training)
                loss = self.mse(y_batch, y_pred, training)
                if type(val_x) != np.ndarray and type(val_y) != np.ndarray:
                    valprint = ""
                else:
                    val_pred = self.predict(val_x, training)
                    val_loss = self.mse(val_y, val_pred, training)
                    valprint = f', validation_loss: {val_loss}'

                if epoch % view_epoch == 0:
                    per = int((batch+1) / (len(x)/batch_size) * 20)
                    if per > 20:
                        per = 20
                    print("▮"*per+"▯"*(20-per),\
                          f"Epoch {epoch}, loss: {loss}" + valprint)
class metrics():
    pass
class callbacks():
    pass
class preprocessing():
    pass
class MatchError(Exception):
    pass
class DataError(Exception):
    pass
        
