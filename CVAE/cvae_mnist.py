import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.misc import imsave

import pickle
import json

path_data = '/home/antorosi/Documents/AutoEncoder/data'
path_out = '/home/antorosi/Documents/AutoEncoder/out'
path_thesis_figures = '/home/antorosi/Documents/Thesis/Figures'

# Loading data set
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

# select optimizer
optim = 'rmsprop'

# parameters
m = 200
n_x = x_train.shape[1]
n_y = y_train.shape[1]
n_z = 2
n_epoch = 30

### Model

# Defining Input
X = Input(shape=(n_x,))
cond = Input(shape=(n_y,))

inputs = concatenate([X, cond])

# Encoder: Q(z|X,y)
h_q = Dense(512, activation='relu')(inputs)
h_q_1 = Dense(256, activation='relu')(h_q)
mu = Dense(n_z, activation='linear')(h_q_1)
log_sigma = Dense(n_z, activation='linear')(h_q_1)


# Sampling from latent space: z ~ Q(z|X,y)
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


z = Lambda(sample_z)([mu, log_sigma])
z_cond = concatenate([z, cond])

# Decoder: P(X|z,y)
decoder_hidden = Dense(512, activation='relu')
decoder_hidden_1 = Dense(256, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden_1(z_cond)
h_p_1 = decoder_hidden(h_p)
outputs = decoder_out(h_p_1)

# Losses
def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z,y)]
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=1)
    # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


def KL_loss(y_true, y_pred):
    return 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)


def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=1)


# Different models:
# Overall VAE model, for reconstruction and training
cvae = Model([X, cond], outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model([X, cond], mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z + n_y,))
d_h = decoder_hidden_1(d_in)
d_h_1 = decoder_hidden(d_h)
d_out = decoder_out(d_h_1)
decoder = Model(d_in, d_out)

# Creating folder for saving
name = 'CVAE_Test_20epochs'
newpath = os.path.join(path_out, name)
if not os.path.exists(newpath):
    os.makedirs(newpath)

# Setting callbacks
callbacks = []

early_stop = EarlyStopping(monitor='val_loss', patience=30)
callbacks.append(early_stop)

model_checkpoint = ModelCheckpoint(os.path.join(newpath, name + "-best.hdf5"),
                                   monitor='val_loss',
                                   verbose=0, save_best_only=True, save_weights_only=False,
                                   mode='auto', period=1)
callbacks.append(model_checkpoint)

# Compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics=[KL_loss, recon_loss])
cvae_hist = cvae.fit([x_train, y_train], x_train, batch_size=m, epochs=n_epoch,
                     validation_data=([x_test, y_test], x_test),
                     callbacks=callbacks, verbose=True)

# Save last model
cvae.save(os.path.join(newpath, name +  '-last_model.hdf5'))
encoder.save(os.path.join(newpath, name +  '-last_encoder.hdf5'))
decoder.save(os.path.join(newpath, name +  '-last_decoder.hdf5'))

# Save history
with open(os.path.join(newpath, name + '-history.json'), 'w') as f:
    json.dump(cvae_hist.history, f)

##################################################################################################################################"
# Loss

def plot_loss():

    plt.plot(cvae_hist.epoch, cvae_hist.history['val_loss'])
    plt.plot(cvae_hist.epoch, cvae_hist.history['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss evolution')
    plt.legend(('test','train'))

    #plt.show()
    plt.savefig(os.path.join(newpath,'loss_evolution.png'))

def plot_projections():
    # Results analysis
    x_test_encoded = encoder.predict([x_test, y_test])
    plt.figure(figsize=(8, 8))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1],s=2,c=y_test_ori, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.xlabel("z1", size=14)
    plt.ylabel("z2", size=14)
    plt.axis("equal")

    name = "MNIST_z2_CVAE"
    plt.savefig(os.path.join(path_thesis_figures, name + '.png'))
    plt.savefig(os.path.join(path_thesis_figures, name + '.pdf'))
    plt.plot()


def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)

def plot_digit():

    dig = 2
    sides = 15
    max_z = 3

    img_it = 0
    for i in range(0, sides):
        z1 = (((i / (sides-1)) * max_z)*2) - max_z
        for j in range(0, sides):
            z2 = (((j / (sides-1)) * max_z)*2) - max_z
            z_ = [z1, z2]
            vec = construct_numvec(dig, z_)
            decoded = decoder.predict(vec)
            plt.subplot(sides, sides, 1 + img_it)
            img_it +=1
            plt.imshow(decoded.reshape(28, 28), cmap = 'gray_r'), plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)

    name = "CVAE_manifold_z2_{}_{}".format(dig,sides)
    plt.savefig(os.path.join(path_thesis_figures, name + '.png'))
    plt.savefig(os.path.join(path_thesis_figures, name + '.pdf'))
    plt.plot()

####

def plot_style():

    ind_list = [150,0,1000, 300, 500,600,700, 800, 900]
    row_nb = len(ind_list)
    img_it = 0

    for ind in ind_list:

        x = x_test[ind,:]
        y = y_test[ind,:]
        c_ori = y_test_ori[ind]

        z = encoder.predict([x.reshape(1,-1),y.reshape(1,-1)]).flatten()
        c_list = [el for el in range(10)]

        plt.subplot(row_nb, 11, 1 + img_it)
        plt.imshow(x.reshape(28, 28), cmap=plt.cm.gray), plt.axis('off')
        img_it +=1

        for i,c in enumerate(c_list):
            vec = construct_numvec(c, z)
            decoded = decoder.predict(vec)
            plt.subplot(row_nb, 11, 1 + img_it)
            img_it += 1
            plt.imshow(decoded.reshape(28, 28), cmap=plt.cm.gray), plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)

    #plt.show()
    plt.savefig(os.path.join(newpath, 'style_generation.png'))

def plot_reconstruction():

    ind=150
    x = x_train[ind,:]
    y = y_train[ind,:]

    # plot one digit
    digit_size = 28
    digit =  x.reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(4, 4))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

    x_recon = cvae.predict([x.reshape(1,-1), y.reshape(1,-1)])
    digit =  x_recon.reshape(digit_size, digit_size)
    figure = digit
    plt.figure(figsize=(4, 4))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

plot_loss()
plot_style()