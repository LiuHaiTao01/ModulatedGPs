import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from scipy.cluster.vq import kmeans2, kmeans

from ModulatedGPs.likelihoods import HeteroGaussian
from ModulatedGPs.models import SHGP
from ModulatedGPs.layers import SVHGP_Layer
from ModulatedGPs.kernels import RBF

from gpflow import settings
float_type = settings.float_type

import matplotlib.pyplot as plt
# %matplotlib inline

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#***************************************
# Load data
#*************************************** 
func = 'hetero'

if func == 'hetero':
    f = lambda X: np.cos(5*X)*np.exp(-X/2)
    g = lambda X: 0.25*(np.cos(6*X)+1)*np.exp(-X)
    N, Ns = 1000, 500
    Xtrain = np.linspace(-2,2,N)[:,None]
    Ytrain = f(Xtrain) + g(Xtrain)*np.random.normal(size=Xtrain.shape)
    Xtest = np.linspace(-2,2,Ns)[:,None]
elif func == 'step':
    N, Ns = 500, 500
    Xtrain = np.linspace(0., 1., N)[:, None]
    Xtest = np.linspace(0., 1., Ns)[:, None]
    f_step = lambda x: 0. if x<0.5 else 1.
    g_step = lambda x: 1e-2
    Ytrain = np.reshape([f_step(x) + np.random.randn() * g_step(x) for x in Xtrain], Xtrain.shape)
elif func == 'moon':
    N, Ns = 200, 500
    noise = 5.0e-2
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=N, shuffle=True, noise=noise)
    Xtrain, Ytrain = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)
    data_test, _ = make_moons(n_samples=Ns, shuffle=True, noise=noise)
    Xtest = np.sort(data_test[:, 0].reshape(-1, 1))

# normalization
Ymean, Ystd = np.mean(Ytrain), np.std(Ytrain)
Ytrain_norm = (Ytrain - Ymean) / Ystd  
Xmean, Xstd = np.mean(Xtrain, axis=0, keepdims=True), np.std(Xtrain, axis=0, keepdims=True)
Xtrain_norm = (Xtrain - Xmean) / Xstd 
Xtest_norm = (Xtest - Xmean) / Xstd

#***************************************
# Model configuration
#***************************************
num_iter            = 10000             # Optimization iterations
lr                  = 5e-3             # Learning rate for Adam solver
num_minibatch       = N                # Batch size for stochastic opt
num_samples         = 1                # Number of MC samples
num_predict_samples = 200              # Number of predict samples
num_data            = Xtrain.shape[0]  # Training size
dimX                = Xtrain.shape[1]  # Input dimensions
dimY                = 1                # Output dimensions
num_ind             = 50               # Inducing size for f 
num_ind_w           = 50               # Inducing size for w

X_placeholder = tf.placeholder(dtype = float_type,shape=[None, dimX])
Y_placeholder = tf.placeholder(dtype = float_type,shape=[None, dimY])
train_dataset  = tf.data.Dataset.from_tensor_slices((X_placeholder,Y_placeholder))
train_dataset  = train_dataset.shuffle(buffer_size=num_data, seed=seed).batch(num_minibatch).repeat()
train_iterator = train_dataset.make_initializable_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
X,Y = iterator.get_next()

m_GP = 'SHGP'

# kernel and inducing points initialization
class KERNEL:
    kern, kern_w = RBF, RBF
    lengthscales, lengthscales_w = 1., 1.
    sf2, sf2_w = 1., 1.
    ARD = True
mu0_w = 0. 

input_dim = dimX
pred_kernel = KERNEL.kern(input_dim=input_dim, lengthscales=KERNEL.lengthscales, variance=KERNEL.sf2, ARD=KERNEL.ARD, name="kernel") 
pred_kernel_w = KERNEL.kern_w(input_dim=input_dim, lengthscales=KERNEL.lengthscales_w, variance=KERNEL.sf2_w, ARD=KERNEL.ARD, name="kernel_w")
Z, Z_w = kmeans(Xtrain_norm,num_ind)[0], kmeans(Xtrain_norm,num_ind_w)[0]
pred_layer = SVHGP_Layer(kern=pred_kernel, kern_g=pred_kernel_w, Z=Z, Z_g=Z_w, mu0_g=mu0_w, num_inducing=num_ind, num_inducing_g=num_ind_w, num_outputs=dimY)
 
# model definition
lik = HeteroGaussian()
model = SHGP(likelihood=lik, pred_layer=pred_layer, num_data=num_data)

#***************************************                 
# Model training
#***************************************
lowerbound = model._build_likelihood(X,Y)
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(-1.*lowerbound)

# prediction ops
samples_y, samples_f = model.predict_samples(X, S=num_predict_samples)
fmean, fvar, wmean, wvar = model.predict_f_w(X)

# tensorflow variable and handle initializations
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

train_handle  = sess.run(train_iterator.string_handle())
sess.run(train_iterator.initializer,{X_placeholder:Xtrain_norm, Y_placeholder:Ytrain_norm})

print('{:>5s}'.format("iter") + '{:>24s}'.format("ELBO:"))
iters = []; elbos = []
for i in range(1,num_iter+1): 
    try:
        sess.run(train_op,feed_dict={handle:train_handle})      
        if i % 100 == 0 or i == 0:           
            elbo = sess.run(lowerbound,{handle:train_handle})  
            print('{:>5d}'.format(i)  + '{:>24.6f}'.format(elbo))
            iters.append(i); elbos.append(elbo)
    except KeyboardInterrupt as e:
        print("stopping training")
        break

#***************************************      
# Prediction and Plot
#***************************************
n_batches = max(int(Xtest_norm.shape[0]/500), 1)
Ss_y, Ss_f = [], []
for X_batch in np.array_split(Xtest_norm, n_batches):
    Ss_y.append(sess.run(samples_y,{X:X_batch})) 
    Ss_f.append(sess.run(samples_f,{X:X_batch}))
samples_y, samples_f = np.hstack(Ss_y), np.hstack(Ss_f)
mu_avg, fmu_avg = np.mean(samples_y, 0), np.mean(samples_f, 0) 
samples_y_stack = np.reshape(samples_y, (num_predict_samples*Xtest_norm.shape[0],-1))
samples_f_stack = np.reshape(samples_f, (num_predict_samples*Xtest_norm.shape[0],-1))
# samples = samples * Ystd + Ymean

f, ax = plt.subplots(2, 2, figsize=(14,8))
Xt_tiled = np.tile(Xtest_norm, [num_predict_samples, 1])
ax[0,0].scatter(Xt_tiled.flatten(), samples_y_stack.flatten(), marker='+', alpha=0.01, color='red')
ax[0,0].scatter(Xt_tiled.flatten(), samples_f_stack.flatten(), marker='+', alpha=0.01, color='blue')
ax[0,0].scatter(Xtrain_norm, Ytrain_norm, marker='x', alpha=0.4, color='black')
ax[0,0].legend([r'samples from $p(y_*|\mathbf{y})$', r'samples from $p(f_*|\mathbf{y})$', 'training data'])
ax[0,0].set_title(m_GP)
ax[0,0].set_xlabel('x') 
ax[0,0].set_ylabel('y')
ax[0,0].set_ylim(1.2*min(Ytrain_norm), 1.2*max(Ytrain_norm))
ax[0,0].grid()

fmean_, fvar_ = sess.run(fmean,{X:Xtest_norm}), sess.run(fvar,{X:Xtest_norm})
wmean_, wvar_ = sess.run(wmean,{X:Xtest_norm}), sess.run(wvar,{X:Xtest_norm})
lb_f, ub_f = (fmean_ - 2*fvar_**0.5), (fmean_ + 2*fvar_**0.5)
lb_w, ub_w = (wmean_ - 2*wvar_**0.5), (wmean_ + 2*wvar_**0.5)
ax[0,1].plot(Xtest_norm, wmean_, '-')
ax[0,1].fill_between(Xtest_norm.flatten(), lb_w.flatten(), ub_w.flatten(), alpha=0.3)
ax[0,1].set_xlabel('x') 
ax[0,1].set_ylabel('w')
ax[0,1].grid()   

ax[1,0].plot(Xtest_norm, fmean_ * np.exp(wmean_), '-')
ax[1,0].plot(Xtest_norm, fmean_, '--')
ax[1,0].scatter(Xtrain_norm, Ytrain_norm, marker='x', color='black', alpha=0.1)
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('latent output') 
ax[1,0].legend([r'$e^w * f$ (modulated output)', r'f'])
ax[1,0].grid()

ax[1,1].plot(Xtest_norm, np.exp(wmean_) * sess.run(lik.c), '-')
ax[1,1].set_xlabel('x') 
ax[1,1].set_ylabel('noise variance')
ax[1,1].grid()

plt.tight_layout()
plt.savefig('figs/'+m_GP+'_'+func+'_toy.png')
plt.show()
