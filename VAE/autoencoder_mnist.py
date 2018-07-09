import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping

import pickle

# Loading data set
path_data ='/home/antorosi/Documents/AutoEncoder/data'
path_thesis_figures = '/home/antorosi/Documents/Thesis/Figures'


with open(os.path.join(path_data,'mnist.pickle'), 'rb') as f:
    mnist = pickle.load(f)

# Model

batch_size = 200
original_dim = 784
latent_dim = 2
intermediate_dim_2 = 512
intermediate_dim = 256
epochs = 30
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim_2, activation='relu')(x)
h = Dense(intermediate_dim, activation='relu')(h)

z = Dense(latent_dim, activation='linear')(h)


# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h_2 = Dense(intermediate_dim_2, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
h_decoded_2 = decoder_h_2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded_2)

# instantiate VAE model
ae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
vae_loss = K.mean(xent_loss)

ae.add_loss(vae_loss)
ae.compile(optimizer='adam')
ae.summary()

# train the VAE on MNIST digits
x_train = mnist['x_train']
y_train = mnist['y_train']
x_test = mnist['x_test']
y_test = mnist['y_test']

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

ae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        callbacks=[EarlyStopping(patience=5)]
        )

# build a model to project inputs on the latent space
encoder = Model(x, z)

def plot_projections():
    # Results analysis
    x_test_encoded = encoder.predict(x_test)
    plt.figure(figsize=(8, 8))
    plt.scatter(x_test_encoded[:, 0], -x_test_encoded[:, 1],s=2,c=y_test, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.xlabel("z1", size=14)
    plt.ylabel("z2", size=14)
    plt.axis('equal')

    name = "MNIST_z2_AE"
    plt.savefig(os.path.join(path_thesis_figures, name + '.png'))
    plt.savefig(os.path.join(path_thesis_figures, name + '.pdf'))
    plt.plot()


# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_h_decoded_2 = decoder_h_2(_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded_2)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(figure, cmap='gray_r')
name = "VAE_manifold_z2_{}".format(n)
plt.savefig(os.path.join(path_thesis_figures, name + '.png'))
plt.savefig(os.path.join(path_thesis_figures, name + '.pdf'))
plt.plot()


#plt.savefig(os.path.join('/home/antorosi/Documents/AutoEncoder/out/figure', 'mnist_manifold,VAE_V0.png'))

