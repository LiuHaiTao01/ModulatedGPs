# Borrowed and modified from: https://github.com/GPflow/GPflow/

from .param import Param

import numpy as np
import tensorflow as tf

from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc


class Likelihood:
    def __init__(self, *args, **kwargs):
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                              self.num_gauss_hermite_points,
                              Fmu, Fvar)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, logspace=True, Y=Y)

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        return ndiagquad(self.logp,
                         self.num_gauss_hermite_points,
                         Fmu, Fvar, Y=Y)


class Gaussian(Likelihood):
    def __init__(self, variance=1e-0, D=None, **kwargs):
        super().__init__(**kwargs)
        if D is not None: # allow different noises for outputs
            variance = variance * np.ones((1,D))
        self.variance = Param(variance, transform=transforms.Log1pe(),name = "noise_variance")()

    def logp(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


class HeteroGaussian(Likelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c = Param(1., transform=transforms.Log1pe(), name="para_c")()

    def predict_mean_and_var(self, Fmu, Fvar, Gmu, Gvar):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Gmu, Gvar, Y):
        Rg_3, Rg_4 = tf.exp(2.*Gvar - 2.*Gmu), tf.exp(0.5 * Gvar - Gmu)
        ve = -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.c) - Gmu \
                - 0.5 * (Y**2 * Rg_3 - 2. * Y * Rg_4 * Fmu + Fmu**2 + Fvar) / self.c
        return ve
