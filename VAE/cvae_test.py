import numpy as np
import os
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


from scipy.misc import imsave

import pickle

# Loading data set
path_data = '/home/antorosi/Documents/AutoEncoder/data'

with open(os.path.join(path_data, 'mnist.pickle'), 'rb') as f:
    mnist = pickle.load(f)

# train the VAE on MNIST digits
x_train = mnist['x_train']
y_train = mnist['y_train']
x_test = mnist['x_test']
y_test = mnist['y_test']

# X normalization
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Y to one hot
y_train_ori = y_train
y_train = np.zeros((y_train_ori.size, y_train_ori.max() + 1))
y_train[np.arange(y_train_ori.size), y_train_ori] = 1

y_test_ori = y_test
y_test = np.zeros((y_test_ori.size, y_test_ori.max() + 1))
y_test[np.arange(y_test_ori.size), y_test_ori] = 1


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 2

# dimension of input (and label)
n_x = x_train.shape[1]
n_y = y_train.shape[1]

# nubmer of epochs
n_epoch = 10

##  ENCODER ##

# encoder inputs
X = Input(shape=(n_x,))
cond = Input(shape=(n_y,))

# merge pixel representation and label
inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape=(n_z,))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##  DECODER  ##

# dense ReLU to sigmoid layers
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')
h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model([X, cond], outputs)
encoder = Model([X, cond], mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z + n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)


# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl


def KL_loss(y_true, y_pred):
    return (0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))


def recon_loss(y_true, y_pred):
    return (K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics=[KL_loss, recon_loss])
cvae_hist = cvae.fit([x_train, y_train], x_train, batch_size=m, epochs=n_epoch,
                     validation_data=([x_test, y_test], x_test),
                     callbacks=[EarlyStopping(patience=5)])

import matplotlib.pyplot as plt

x_test_encoded = encoder.predict([x_train, y_train])
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_train_ori, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

ind = 200
x = x_train[ind, :]
y = y_train[ind, :]

# plot one digit
digit_size = 28
digit = x.reshape(digit_size, digit_size)
figure = digit
plt.figure(figsize=(4, 4))
plt.imshow(figure, cmap='Greys_r')
plt.show()

x_recon = cvae.predict([x.reshape(1, -1), y.reshape(1, -1)])
digit = x_recon.reshape(digit_size, digit_size)
figure = digit
plt.figure(figsize=(4, 4))
plt.imshow(figure, cmap='Greys_r')
plt.show()
