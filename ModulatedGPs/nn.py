import numpy as np
import tensorflow as tf
from .param import Param
from gpflow import settings
from gpflow import transforms
from scipy.stats import truncnorm

float_type = settings.float_type
jitter_level = settings.jitter

def xavier_initializer(input_dim, output_dim):
    xavier_std = (2. / (input_dim + output_dim)) ** 0.5
    return xavier_std

def he_initializer(input_dim, output_dim):
    return np.sqrt(2. / input_dim)

class mlp_share:
    '''
    3-hidden-layer MLPs shared for mean and variance
    '''
    def __init__(self,input_dim, output_dim, inner_dims=None, activation=None, initializer=None, var=None, name="encoder_share"):
        self.input_dim = input_dim
        self.output_dim = output_dim # output_dim = 2 * input_dim for mean and variance
        if inner_dims is None: # three hidden layers
            dim = max(input_dim * 2, 10)
            inner_dims = [dim, dim, dim]
        self.inner_dims = inner_dims
        self.layer_dims = [input_dim, *inner_dims, output_dim] # three hidden layers
        self.name = name
        self.activation = activation or tf.nn.relu
        self.initializer = initializer or xavier_initializer
        Ws, bs = [], []
        with tf.name_scope(self.name):
            for i, (n_in, n_out) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
                init_scale = self.initializer(n_in, n_out)
                Ws.append(Param(truncnorm.rvs(-2, 2, size=(n_in, n_out)) * init_scale, name="w_"+str(i+1))())
                bs.append(Param(np.zeros((1,n_out)), name="b_"+str(i+1))())
        self.Ws, self.bs = Ws, bs
        self.var = var

    def forward(self, y): # t is useless
        y_ = tf.identity(y)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            y_ = tf.matmul(y_, W) + b
            if i < len(self.layer_dims) - 2:
                y_ = self.activation(y_)
        y_mean, y_var = tf.split(y_, 2, axis=-1)
        if self.var is None:
            y_var = tf.nn.softplus(y_var - 5.)
        else:
            y_var = self.var * tf.sigmoid(y_var + 0.)
        return y_mean, y_var

