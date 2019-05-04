from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch.nn import GRUCell
from torch.nn import Linear

import time

from lib.metrics import masked_mae_loss

torch.set_default_tensor_type(torch.FloatTensor)

class MPNNModel(torch.nn.Module):
    def __init__(self, is_training, batch_size, message_size, scaler, adj_mx, **model_kwargs):
        # Scaler for data normalization.
        super(MPNNModel,self).__init__()
        self._scaler = scaler
        self._horizon = int(model_kwargs.get('horizon', 1))
        self._num_nodes = int(model_kwargs.get('num_nodes', 1))
        self._rnn_units = int(model_kwargs.get('rnn_units'))
        self._input_dim = int(model_kwargs.get('input_dim', 1))
        self._output_dim = int(model_kwargs.get('output_dim', 1))
        self._message_size = message_size
        self._batch_size = batch_size
        self._adj_mx = adj_mx
        self._cell = GRUCell((self._input_dim+self._message_size),self._rnn_units)
        self._M_t = Linear(self._rnn_units,self._message_size)
        self._R_t = Linear(self._rnn_units,self._output_dim*self._horizon)

    def forward(self,inputs):
        inps = torch.unbind(inputs,dim=1)
        h = torch.randn(self._num_nodes,self._batch_size,self._rnn_units)
        for inp in inps:
            h_w_prod = self._M_t(h)
            m = torch.zeros(self._num_nodes,self._batch_size,self._message_size)
            x0 = inp.permute(1,0,2)
            for i in range(self._num_nodes):
                for j in range(self._num_nodes):
                    if self._adj_mx[j,i] > 0.0:
                      m[i] = m[i] + np.asscalar(self._adj_mx[j,i])*h_w_prod[j]

            inp_2 = torch.cat((m,x0),dim = 2)
            h = h.permute(1,0,2)
            h = h.reshape(self._batch_size*self._num_nodes,-1)
            inp_2 = inp_2.permute(1,0,2)
            inp_2 = inp_2.reshape(self._batch_size*self._num_nodes,-1)
            h = self._cell(inp_2,h)
            h = h.reshape(self._batch_size,self._num_nodes,self._rnn_units)
            h = h.permute(1,0,2)

        outputs = self._R_t(h)
        outputs = outputs.reshape(self._num_nodes,self._batch_size,self._horizon,-1).permute(1,2,0,3)
        return outputs
