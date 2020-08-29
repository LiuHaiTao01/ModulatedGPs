from .param import Param
from .utils import reparameterize
from .likelihoods import Gaussian
from .broadcasting_lik import BroadcastingLikelihood, BroadcastingLikelihood_HGP
from .nn import mlp_share
from .utils import pca

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from gpflow import settings
from gpflow import transforms
from .settings import Settings
float_type = settings.float_type
jitter_level = settings.jitter


class SGP:
    """
    Scalable Gaussian process (SGP)
    X -> Xt = integrate(X) -> GP -> Y
    """
    def __init__(self, likelihood, pred_layer, num_samples=1, num_data=None):
        self.num_samples = num_samples
        self.num_data = num_data
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.pred_layer = pred_layer

    def integrate(self, X, S=1): 
        return tf.tile(X[None, :, :], [S,1,1]), None 

    def propagate(self, Xt, full_cov=False): 
        F, Fmean, Fvar = self.pred_layer.sample_from_conditional(Xt, full_cov=full_cov)
        return F, Fmean, Fvar 

    def _build_predict(self, Xt, full_cov=False): 
        _, Fmeans, Fvars = self.propagate(Xt, full_cov=full_cov)
        return Fmeans, Fvars 

    def E_log_p_Y(self, Xt, Y): 
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  
        return tf.reduce_mean(tf.reduce_sum(var_exp, 2), 0)  

    def _build_likelihood(self,X,Y): 
        Xt = self.integrate(X, self.num_samples)[0] 
        L = tf.reduce_mean(self.E_log_p_Y(Xt,Y))
        return L - self.pred_layer.KL() / self.num_data 

    def predict_f(self, Xnew, S=1): 
        Xnewt = self.integrate(Xnew, S)[0]
        return self._build_predict(Xnewt, full_cov=False)

    def predict_y(self, Xnew, S=1): 
        Xnewt = self.integrate(Xnew, S)[0]
        Fmean, Fvar = self._build_predict(Xnewt, full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_samples(self, Xnew, S=1): 
        Fmean, Fvar = self.predict_f(Xnew, S)
        mean, var = self.likelihood.predict_mean_and_var(Fmean, Fvar)
        z = tf.random_normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_f = reparameterize(Fmean, Fvar, z)
        return samples_y, samples_f 

    def predict_density(self, Xnew, Ynew, S):
        Fmean, Var = self.predict_y(Xnew, S)
        l = self.likelihood.predict_density(Fmean, Var, Ynew)
        log_num_samples = tf.log(tf.cast(S, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    def get_inducing_Z(self):
        return self.pred_layer.Z


class SMGP(SGP):
    '''
    Mixture of Gaussian processes, used for regression, density estimation, data association, etc
    '''
    def __init__(self, likelihood, pred_layer, assign_layer, K=3, num_samples=1, num_data=None):
        SGP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.assign_layer = assign_layer 
        self.K = K 

    def propagate_logassign(self, Xt, full_cov=False): 
        logassign, logassign_mean, logassign_var = self.assign_layer.sample_from_conditional(Xt, full_cov=full_cov)
        return logassign, logassign_mean, logassign_var 

    def _build_predict_logassign(self, Xt, full_cov=False): 
        _, logassign_mean, logassign_var = self.propagate_logassign(Xt, full_cov=full_cov)
        return logassign_mean, logassign_var 

    def W_dist(self, Xt):
        logassign_mean, logassign_var = self._build_predict_logassign(Xt) 
        z = tf.random_normal(tf.shape(logassign_mean), dtype=float_type)
        log_assign = reparameterize(logassign_mean, logassign_var, z) 
        log_assign = tf.reshape(log_assign, [tf.shape(Xt)[0]*tf.shape(Xt)[1], self.K]) 
        W_dist = tfp.distributions.RelaxedOneHotCategorical(temperature=1e-2, logits=log_assign)
        return W_dist

    def E_log_p_Y(self, Xt, Y, W_SND): 
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  
        var_exp *= tf.cast(W_SND, dtype=float_type)
        return tf.reduce_logsumexp(tf.reduce_sum(var_exp, 2), 0) - np.log(self.num_samples) 

    def _build_likelihood(self,X,Y):
        Xt = self.integrate(X, self.num_samples)[0] 
        # sample from q(w)
        W_dist = self.W_dist(Xt)
        W = W_dist.sample(1)[0,:,:] 
        W_SND = tf.reshape(W, [self.num_samples, tf.shape(Xt)[1], self.K]) 
        # Expectation of lik
        L = tf.reduce_mean(self.E_log_p_Y(Xt, Y, W_SND))
        # ELBO
        return L - (self.pred_layer.KL() + self.assign_layer.KL()) / self.num_data 
    
    def predict_assign(self, Xnew, S=1):
        Xt = self.integrate(Xnew, S)[0] 
        logassign_mean, logassign_var = self._build_predict_logassign(Xt) 
        assign = tf.nn.softmax(tf.exp(tf.reduce_mean(logassign_mean,0))) 
        return assign

    def predict_samples(self, Xnew, S=1): 
        Xt = self.integrate(Xnew, S)[0] 
        W_dist = self.W_dist(Xt)
        W = W_dist.sample(1)[0,:,:] 
        W_SND = tf.cast(tf.reshape(W, [S, tf.shape(Xt)[1], self.K]), dtype=float_type) 
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        mean, var = self.likelihood.predict_mean_and_var(Fmean, Fvar)
        z = tf.random_normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z) 
        samples_y = tf.reduce_sum(samples_y * W_SND, 2, keepdims=True)
        samples_f = reparameterize(Fmean, Fvar, z) 
        samples_f = tf.reduce_sum(samples_f * W_SND, 2, keepdims=True)
        return samples_y, samples_f 


class SHGP(SGP):
    '''
    HGPs for regression, including
    Heteroscedastic GP, non-stationary GP, and non-stationary heteroscedastic GP
    '''
    def __init__(self, likelihood, pred_layer, num_data=None,):
        self.num_data = num_data
        self.lik = likelihood
        self.likelihood = BroadcastingLikelihood_HGP(likelihood)
        self.pred_layer = pred_layer

    def propagate(self, Xt, full_cov=False):
        F, W, Fmean, Fvar, Wmean, Wvar = self.pred_layer.sample_from_conditional(Xt, full_cov=full_cov)
        return F, W, Fmean, Fvar, Wmean, Wvar

    def _build_predict(self, Xt, full_cov=False):
        Fs, Ws, Fmeans, Fvars, Wmeans, Wvars = self.propagate(Xt, full_cov=full_cov)
        return Fmeans, Fvars, Wmeans, Wvars

    def E_log_p_Y(self, Xt, Y):
        Fmean, Fvar, Wmean, Wvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Wmean, Wvar, Y)  
        return tf.reduce_mean(tf.reduce_sum(var_exp, 2), 0)

    def _build_likelihood(self,X,Y):
        Xt = self.integrate(X)[0]
        # E_{q(f)q(w)}[logp(y|f,w)]
        L = tf.reduce_mean(self.E_log_p_Y(Xt,Y))
        # ELBO
        return L - (self.pred_layer.KL() + self.pred_layer.KL_g()) / self.num_data

    def predict_samples(self, Xnew, S=1):
        Xnewt = self.integrate(Xnew, S)[0]
        Fmean, Fvar, Wmean, Wvar = self._build_predict(Xnewt, full_cov=False)
        z_w = tf.random_normal(tf.shape(Wmean), dtype=float_type)
        samples_w = reparameterize(Wmean, Wvar, z_w)
        mean, var = Fmean * tf.exp(samples_w), Fvar * tf.exp(2.*samples_w) + self.lik.c * tf.exp(2.*samples_w)
        z = tf.random_normal(tf.shape(mean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_f = reparameterize(mean, Fvar*tf.exp(2.*samples_w), z)
        return samples_y, samples_f
    
    def predict_f_w(self, Xnew, S=1):
        Xnewt = self.integrate(Xnew, S=S)[0]
        Fmean, Fvar, Wmean, Wvar = self._build_predict(Xnewt, full_cov=False)
        return tf.reduce_mean(Fmean,0), tf.reduce_mean(Fvar,0), tf.reduce_mean(Wmean,0), tf.reduce_mean(Wvar,0)    

    def get_inducing_Z_w(self):
        return self.pred_layer.Z_g


class LGP_base(SGP):
    """
    Latent Gaussian process
    """
    def __init__(self, likelihood, pred_layer, dimX, dimY, dimW, 
                 num_samples=1, num_data=None, elbo='IWVI-VI', inner_dims=[100,100,100]):
        SGP.__init__(self, likelihood, pred_layer, num_samples, num_data)
        self.lik = likelihood 
        self.dimX, self.dimY, self.dimW = dimX, dimY, dimW
        self.inner_dims = inner_dims
        self.nn_W_xy = mlp_share(self.dimX+self.dimY, 2*self.dimW, inner_dims=inner_dims, name="encoder_w_xy")
        self.nn_W_x = mlp_share(self.dimX, self.dimW * 2, inner_dims=inner_dims, name="prior_w_x")
        self.method = elbo

    def encoder_W(self, X, Y=None, inference=True, S=1):
        if inference is True: # q(w|x,y)
            XY = tf.concat([X, Y], -1)
            W_mean, W_var = self.nn_W_xy.forward(XY) 
        else: # p(w|x)
            W_mean, W_var = self.nn_W_x.forward(X)
        sW_mean, sW_var = tf.tile(W_mean[None,:,:], [S,1,1]), tf.tile(W_var[None,:,:], [S,1,1])
        z = tf.random_normal(tf.shape(sW_mean), dtype=float_type)
        W = reparameterize(sW_mean, sW_var, z) # S,N,D
        X_tiled = tf.tile(X[None,:,:], [S,1,1])
        X_W = tf.concat([X_tiled, W], -1)     
        return X_W, W_mean, W_var, W
        
    def integrate(self, X, Y=None, inference=True, S=1):
        return self.encoder_W(X, Y, inference, S)

    def E_log_p_Y(self, Xt, Y, p_w_x=None, q_w_xy=None, Ws=None): 
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  
        if self.method == 'VI':
            return tf.reduce_mean(tf.reduce_sum(var_exp, 2), 0)  
        elif self.method == 'IWVI':
            local_w_logp = lambda w: p_w_x.log_prob(w) - q_w_xy.log_prob(w)  
            local_w_logps = tf.map_fn(local_w_logp, Ws, dtype=float_type) 
            return tf.reduce_logsumexp(tf.reduce_sum(var_exp,2)+tf.reduce_sum(local_w_logps,2), 0) - np.log(self.num_samples) # N,
        elif self.method == 'IWVI-VI':
            raise NotImplementedError

    def _build_likelihood(self,X,Y):
        Xt, W_xy_mean, W_xy_var, W = self.integrate(X, Y=Y, S=self.num_samples)
        # Expectation of lik
        W_x_mean, W_x_var = self.nn_W_x.forward(X)      
        p_w_x = tf.distributions.Normal(loc=W_x_mean, scale=tf.sqrt(W_x_var+1e-6))
        q_w_xy = tf.distributions.Normal(loc=W_xy_mean, scale=tf.sqrt(W_xy_var+1e-6))
        if self.method == 'VI':
            L = tf.reduce_mean(self.E_log_p_Y(Xt, Y))
            KL_W = tf.reduce_mean(tf.reduce_sum(q_w_xy.kl_divergence(p_w_x),-1))
            return L - KL_W - self.pred_layer.KL() / self.num_data
        elif self.method == 'IWVI':
            L = tf.reduce_mean(self.E_log_p_Y(Xt, Y, p_w_x, q_w_xy, W)) 
            return L - self.pred_layer.KL() / self.num_data
        elif self.method == 'IWVI-VI':
            raise NotImplementedError

    def predict_samples(self, Xnew, S=1): 
        Xnewt = self.integrate(Xnew, inference=False, S=S)[0] 
        Fmean, Fvar = self._build_predict(Xnewt, full_cov=False) 
        mean, var = self.likelihood.predict_mean_and_var(Fmean, Fvar)
        z = tf.random_normal(tf.shape(Fmean), dtype=float_type)
        samples_y = reparameterize(mean, var, z)
        samples_f = reparameterize(Fmean, Fvar, z)
        return samples_y, samples_f 

    def reconstruct_samples(self, X, Y, S=1): 
        Xt = self.integrate(X, Y, inference=True, S=S)[0] 
        Fmean, _ = self._build_predict(Xt, full_cov=False) 
        return tf.reduce_mean(Fmean, 0) 

    def get_W_XY(self, X, Y):
        XY = tf.concat([X, Y], -1)
        W_mean, W_var = self.nn_W_xy.forward(XY) 
        return W_mean, W_var

    def get_W_X(self, X):
        W_mean, W_var = self.nn_W_x.forward(X) 
        return W_mean, W_var


def get_encoder_prior(X, dim_in, dim_out):
    if dim_in == dim_out:
        Z_prior = tf.identity(X)
    elif dim_in > dim_out: # PCA projection
        Z_prior = pca(X, dim=dim_out)
    else: # zero-padding
        Z_prior = tf.concat([X, tf.zeros([tf.shape(X)[0], dim_out-dim_in], dtype=float_type)], axis=-1)
    return Z_prior

class SLGP(LGP_base):
    '''
    A stochastic MLP transformation applied to (X,W): [H_mean, H_var] = MLP(X,W)
    '''
    def __init__(self, likelihood, pred_layer, dimX, dimY, dimW, latent_dim, Zx, Zw, 
                 num_samples=1, num_data=None, elbo='IWVI-VI', beta=1e-2, inner_dims=[100,100,100]):
        LGP_base.__init__(self, likelihood, pred_layer, dimX, dimY, dimW, num_samples, num_data, elbo, inner_dims)
        self.latent_dim = latent_dim
        self.beta = beta
        self.variance = Param(1e-2, transform=transforms.Log1pe(), name="prior_var")()
        # reassign Z
        self.Zx, self.Zw = Param(Zx, name="zx")(), Param(Zw, name="zw")()
        self.Zw_latent = self.nn_W_xy.forward(self.Zw)[0]
        self.nn_X_xw = mlp_share(self.dimX + self.dimW, self.latent_dim*2, inner_dims=inner_dims, var=self.variance)
        Z_trans = self.nn_X_xw.forward(tf.concat([self.Zx, self.Zw_latent], -1))[0]
        self.pred_layer.initialize_Z(Z_trans) # convert Z to tensor

    def integrate(self, X, Y=None, inference=True, S=1):
        X_W, W_mean, W_var, W = self.encoder_W(X, Y, inference, S) 
        X_W_flat = tf.reshape(X_W, [S * tf.shape(X_W)[1], tf.shape(X_W)[2]]) 
        H_mean, H_var = self.nn_X_xw.forward(X_W_flat) 
        H_mean, H_var = tf.reshape(H_mean, [S, tf.shape(X)[0], self.latent_dim]), tf.reshape(H_var, [S, tf.shape(X)[0], self.latent_dim]) #S,N,D
        z = tf.random_normal(tf.shape(H_mean), dtype=float_type) 
        Xt = reparameterize(H_mean, H_var, z) 
        return Xt, W_mean, W_var, H_mean, H_var, X_W, W

    def E_log_p_Y(self, Xt, Y, p_h_w=None, q_h_w=None, p_w_x=None, q_w_xy=None, Ws=None):
        Fmean, Fvar = self._build_predict(Xt, full_cov=False)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  
        if self.method == 'VI':
            return tf.reduce_mean(tf.reduce_sum(var_exp, 2), 0)  
        elif self.method == 'IWVI':
            local_w_logp = lambda w: tf.log((p_w_x.prob(w) + 1e-6) / (q_w_xy.prob(w) + 1e-6))
            local_w_logps = tf.map_fn(local_w_logp, Ws, dtype=float_type) 
            local_h_logps = tf.log((p_h_w.prob(Xt) + 1e-6) / (q_h_w.prob(Xt) + 1e-6))
            return tf.reduce_logsumexp(tf.reduce_sum(var_exp,2)+tf.reduce_sum(local_w_logps,2)+self.beta*tf.reduce_sum(local_h_logps,2), 0) - np.log(self.num_samples)
        elif self.method == 'IWVI-VI': # hybrid of IWVI and VI
            local_w_logp = lambda w: tf.log((p_w_x.prob(w) + 1e-6) / (q_w_xy.prob(w) + 1e-6))
            local_w_logps = tf.map_fn(local_w_logp, Ws, dtype=float_type) 
            return tf.reduce_logsumexp(tf.reduce_sum(var_exp,2)+tf.reduce_sum(local_w_logps,2), 0) - np.log(self.num_samples)            

    def _build_likelihood(self,X,Y): 
        Xt, W_xy_mean, W_xy_var, H_mean, H_var, X_W, W = self.integrate(X, Y=Y, S=self.num_samples)
        # Expectation of lik
        W_x_mean, W_x_var = self.nn_W_x.forward(X) # N,D p(w|x) 
        p_w_x = tf.distributions.Normal(loc=W_x_mean, scale=tf.sqrt(W_x_var+1e-6))
        q_w_xy = tf.distributions.Normal(loc=W_xy_mean, scale=tf.sqrt(W_xy_var+1e-6))
        
        prior_xw = lambda xw: get_encoder_prior(xw,self.dimX+self.dimW,self.latent_dim)
        p_h_w = tf.distributions.Normal(loc=tf.map_fn(prior_xw, X_W), scale=tf.ones_like(Xt) * tf.sqrt(self.variance+1e-6))
        q_h_w = tf.distributions.Normal(loc=H_mean, scale=tf.sqrt(H_var+1e-6)) 
        if self.method == 'VI':
            L = tf.reduce_mean(self.E_log_p_Y(Xt, Y))
            KL_W = tf.reduce_mean(tf.reduce_sum(q_w_xy.kl_divergence(p_w_x),-1))
            KL_H = tf.reduce_mean( tf.reduce_sum(q_h_w.kl_divergence(p_h_w),-1) )
            # ELBO           
            return L - self.beta * KL_H - KL_W - self.pred_layer.KL() / self.num_data
        elif self.method == 'IWVI':
            L = tf.reduce_mean(self.E_log_p_Y(Xt, Y, p_h_w, q_h_w, p_w_x, q_w_xy, W)) 
            return L - self.pred_layer.KL() / self.num_data
        elif self.method == 'IWVI-VI':
            L = tf.reduce_mean(self.E_log_p_Y(Xt, Y, p_h_w, q_h_w, p_w_x, q_w_xy, W)) 
            KL_H = tf.reduce_mean( tf.reduce_sum(q_h_w.kl_divergence(p_h_w),-1) )
            return L - self.beta * KL_H - self.pred_layer.KL() / self.num_data


