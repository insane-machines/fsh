import numpy as np
from pandas import v
from fsh.addons.preprocess import preprocessing
from fsh.main_addons.std_metrics import metrics
from fsh.errors.errors import DataError, MatchError, ProcessError

class Linear():
    def __init__(self, n_features=1, learning_rate = 0.001):
        self.learning_rate  = learning_rate
        self.weight         = np.random.randn(n_features, 1)
        self.bias           = 0
        self.n_features     = n_features
        self.stop_training  = False
        
    def predict(self, x, training=False):
        y_pred = x @ self.weight + self.bias
        if not training:
            print(f'▶ Predicted output for x = {x}: {y_pred}')
        return y_pred

    def calculate_gradients(self, x, y, training=True):
        error = y - self.predict(x, training=training)
        n = x.shape[0]
        weight_gradient = -2 / n * (x.T @ error)
        bias_gradient = -2 / n * np.sum(error)

        return weight_gradient, bias_gradient

    def train(self, x, y, val_x=None, val_y=None, epochs=50, loss_function=metrics.mse, training=True, callbacks=None, num_epochs_view=1, batch_size=1):
        x = preprocessing.to_array(x)
        y = preprocessing.to_array(y)

        if val_x is not None and val_y is not None:
            val_x = preprocessing.to_array(val_x)
            val_y = preprocessing.to_array(val_y)

        print('## FSH: training starting ##')
        #MAIN CYCLE
        for epoch in range(1, epochs + 1):
            if type(x) != np.ndarray or type(y) != np.ndarray:
                raise DataError(f'TypeError: Invalid type of training data: needs <np.ndarray>')

            if len(x) != len(y):
                raise MatchError('Sizes of datas is not match')

            x_data = np.split(x, np.arange(batch_size, len(x), batch_size))
            y_data = np.split(y, np.arange(batch_size, len(y), batch_size))
            for current_batch_num, x_batch, y_batch in enumerate(zip(x_data, y_data)):
                weight_gradient, bias_gradient = self.calculate_gradients(x_batch, y_batch, training=training)
            
                self.weight -= self.learning_rate * weight_gradient
                self.bias   -= self.learning_rate * bias_gradient

                y_pred = self.predict(x_batch, training)
                loss = loss_function(y_batch, y_pred, training)
                if loss is None:
                    raise ProcessError(f'detected None on epoch {epoch}, batch {current_batch_num}')

                if val_x is not None and val_y is not None:
                    val_pred = self.predict(val_x, training)
                    val_loss = loss_function(val_y, val_pred, training)
                    validation_print = f', validation_loss: {val_loss}'
                else:
                    validation_print = ""
                    val_loss = None

                if epoch % num_epochs_view == 0:
                    total_batches = (len(x) + batch_size - 1) // batch_size
                    batch_completed_percentage = (current_batch_num + 1) / total_batches 
                    visual_progress = int(batch_completed_percentage * 20)
                    visual_progress = min(visual_progress, 20)  

                    print("▮" * visual_progress + "▯" * (20 - visual_progress),\
                         f"{visual_progress * len(x)}/{len(x)} Epoch {epoch}, loss: {loss}" + validation_print)

            logs = {'epoch': epoch, 'loss': loss, 'val_loss': val_loss}
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, logs)
            
            if self.stop_training:
                print(f'FSH: Early stopped at the epoch {epoch}')
                break

        print('## FSH: training completed ##')