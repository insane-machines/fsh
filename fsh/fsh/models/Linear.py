from math import nan, sqrt
import numpy as np
from fsh.addons.preprocess import preprocessing
from fsh.main_addons.std_metrics import metrics
from fsh.errors.errors import DataError, MatchError, ProcessError

class Linear():
    def __init__(self, n_features=1, lr = 0.001):
        self.lr         = lr
        self.weight     = np.random.randn(n_features, 1)
        self.bias       = 0
        self.n_features = n_features
        self.stop = False
    def predict(self, x, training=False):
        y_pred = x @ self.weight + self.bias
        if not training:
            print(f'▶ Predicted output for x = {x}: {y_pred}')
        return y_pred

    def calc_grad(self, x, y, training=True):
        error = y - self.predict(x, training=training)
        n = x.shape[0]
        grad_w = -2 / n * (x.T @ error)
        grad_b = -2 / n * np.sum(error)

        return grad_w, grad_b

    def train(self, x, y, val_x=None, val_y=None, epochs=50, loss_fn=None, training=True, callbacks=None, view_epoch=1, batch_size=1):
        x = preprocessing.to_array(x)
        y = preprocessing.to_array(y)

        if val_x is not None and val_y is not None:
            val_x = preprocessing.to_array(val_x)
            val_y = preprocessing.to_array(val_y)
        
        if loss_fn is None:
            loss_fn = metrics.mse

        print('## FSH: training starting ##')
        #MAIN CYCLE
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

                grad_w, grad_b = self.calc_grad(x_batch, y_batch, training=training)
            
                self.weight -= self.lr * grad_w
                self.bias   -= self.lr * grad_b

                y_pred = self.predict(x_batch, training)
                loss = loss_fn(y_batch, y_pred, training)
                if loss is None:
                    raise ProcessError(f'detected None on epoch {epoch}, batch {batch}')

                if val_x is None and val_y is None:
                    valprint = ""
                    val_loss = None
                else:
                    val_pred = self.predict(val_x, training)
                    val_loss = loss_fn(val_y, val_pred, training)
                    valprint = f', validation_loss: {val_loss}'

                if epoch % view_epoch == 0:
                    total_batches = (len(x) + batch_size - 1) // batch_size
                    batch_per = (batch + 1) / total_batches 
                    per = int(batch_per * 20)
                    per = min(per, 20)  

                    print("▮" * per + "▯" * (20 - per), f"{batch_per*len(x)}/{len(x)} Epoch {epoch}, loss: {loss}" + valprint)

            logs = {'epoch': epoch, 'loss': loss, 'val_loss': val_loss}
            if callbacks is not None:
                for cb in callbacks:
                    cb(self, logs)
            
            if self.stop:
                print(f'FSH: Early stopped at the epoch {epoch}')
                break
        print('## FSH: training stopped ##')