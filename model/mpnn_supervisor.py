from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import torch
import time
import yaml

from lib import utils, metrics
from lib.metrics import masked_mae_loss

from model.mpnn_model import MPNNModel

torch.set_default_tensor_type(torch.FloatTensor)

class MPNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')


        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                print((k, v.shape))

        # Build models.
        scaler = self._data['scaler']
        self._train_model = MPNNModel(is_training=True, scaler=scaler,
                                       batch_size=self._data_kwargs['batch_size'],
                                       message_size = 16,
                                       adj_mx=adj_mx, **self._model_kwargs)

        self._test_model = MPNNModel(is_training=False, scaler=scaler,
                                      batch_size=self._data_kwargs['test_batch_size'],
                                      message_size = 16,
                                      adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._base_lr = 0.01

        # Configure optimizer
        self._optimizer = torch.optim.Adam(lr=1e-2,params=self._train_model.parameters())

        # Calculate loss
        output_dim = self._model_kwargs.get('output_dim')

        null_val = 0.
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._epoch = 0

    def train(self, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10):
        history = []
        min_val_loss = float('inf')
        wait = 0



        while self._epoch <= epochs:
            new_lr = max(min_learning_rate, self._base_lr * (lr_decay_ratio ** (int(self._epoch/10))))
            data_generator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            for _,(x,y) in enumerate(data_generator):
                x = torch.from_numpy(x).type(torch.FloatTensor)
                self._train_model.zero_grad()
                outputs = self._train_model(x)
                loss = self._loss_fn(outputs,y)
                losses.append(np.asscalar(loss))
                print(loss)
                loss.backward()
                self._optimizer.step()

            loss = np.mean(losses)
            print('Epoch : '+str(self._epoch)+' Training Loss : '+str(loss))

            data_generator = self._data['val_loader'].get_iterator()

            losses = []

            for _,(x,y) in enumerate(data_generator):
                x = torch.from_numpy(x).type(torch.FloatTensor)
                outputs = self._train_model(x)
                loss = self._loss_fn(outputs,y)
                losses.append(np.asscalar(loss))
                print(loss)

            loss = np.mean(losses)
            print('Epoch : '+str(self._epoch)+' validation Loss : '+str(loss))
            end_time = time.time()
            print(end_time-start_time)
            if val_loss <= min_val_loss:
                wait = 0
                model_filename = torch.save(self._train_model.state_dict(),'model_val_loss'+str(val_loss))
                print('Val loss decrease from %.4f to %.4f' % (min_val_loss, val_loss))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    print('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

        return np.min(history)
