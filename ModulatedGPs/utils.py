import tensorflow as tf
import numpy as np
from gpflow import settings
float_type = settings.float_type

def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,U,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5
    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None,None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_res = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_res)[:, :,:, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND

def pca(x, dim = 2):
    '''
    PCA for dimensionality reduction 
    '''
    m, n = tf.shape(x)[0], tf.shape(x)[1]
    mean = tf.reduce_mean(x, axis=1)
    x_new = x - tf.reshape(mean,(-1,1)) 
    cov = tf.matmul(x_new, x_new, transpose_a=True) / tf.cast(m - 1, dtype=float_type) 
    e, v = tf.linalg.eigh(cov, name="eigh") 
    e_index_sort = tf.math.top_k(e, sorted=True, k=dim)[1] 
    v_new = tf.gather(v, indices=e_index_sort) 
    pca = tf.matmul(x_new, v_new, transpose_b=True) 
    return pca