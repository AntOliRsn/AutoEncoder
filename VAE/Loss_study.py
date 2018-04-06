import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

z_mean = K.variable(np.array([1,2]))
z_log_var = K.variable(np.array([1,2]))
z_log_sigma = z_log_var

x = K.variable(np.random.rand(4,784))
x_decoded_mean = K.variable(np.random.rand(4,784))

z_mean = K.variable(np.random.rand(10,20))
z_log_sigma = K.variable(np.random.rand(10,20))




# Loss 1
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std, seed=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


def loss(x, x_decoded_mean, z_mean, z_log_var):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    return K.eval(vae_loss)


def vae_loss(x, x_decoded_mean, z_mean, z_log_sigma):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis = 1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=-1)

    return K.eval(recon + kl)


loss = K.sum(K.mean(K.square(x - x_decoded_mean), axis=-1))
K.eval(loss)

K.eval(K.mean(K.square(x - x_decoded_mean), axis=-1))

kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma, axis=-1)
K.eval(kl)