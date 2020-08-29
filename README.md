



Modulating Scalable Gaussian Processes for Expressive Statistical Learning
====

This is the python implementation of scalable modulated Gaussian processes.

For a learning task, Gaussian process (GP) is interested in learning the statistical relationship between inputs and outputs, since it offers not only the prediction mean but also the associated variability. The vanilla GP however struggles to learn complicated distribution with the property of, e.g., **heteroscedastic noise**, **multi-modality** and **non-stationarity**, from **massive data** due to the Gaussian marginal and the cubic complexity. To this end, this article studies new scalable GP paradigms including the *non-stationary heteroscedastic GP*, *the mixture of GPs* and *the latent GP*, which introduce additional latent variables to modulate the outputs or inputs in order to learn richer, non-Gaussian statistical representation. We further resort to different variational inference strategies to arrive at **analytical** or **tighter** evidence lower bounds (ELBOs) of the marginal likelihood for efficient and effective model training. Extensive numerical experiments against state-of-the-art GP and neural network (NN) counterparts on various tasks verify the superiority of these scalable modulated GPs, especially the scalable latent GP, for learning diverse data distributions.

The model is implemented based on [GPflow 1.3.0](https://github.com/GPflow/GPflow) and tested using Tensorflow 1.15.0. 

The demos of three scalable modulated GPs (SHGP, SMGP and SLGP) on three toy cases with heteroscedastic noise, step and multi-modal behaviors are provided. The readers can run the demo files as

```
python demo_SHGP_toy.py
python demo_SMGP_toy.py
python demo_SLGP_toy.py
```