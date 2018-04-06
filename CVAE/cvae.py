import os
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import json

def compile_cvae(dim_input = 96, dim_cond = 24, dim_latent_space = 2, optim = 'rmsprop', verbose = True):

    # parameters
    n_x = dim_input
    n_y = dim_cond
    n_z = dim_latent_space

    ### Model

    # Defining Input
    X = Input(shape=(n_x,))
    cond = Input(shape=(n_y,))

    inputs = concatenate([X, cond])

    # Encoder: Q(z|X,y)
    h_q = Dense(24*2, activation='relu')(inputs)
    h_q_1 = Dense(24, activation='relu')(h_q)
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

    decoder_hidden_1 = Dense(24, activation='relu')
    decoder_hidden = Dense(24*2, activation='relu')
    decoder_out = Dense(dim_input, activation='linear')

    h_p = decoder_hidden_1(z_cond)
    h_p_1 = decoder_hidden(h_p)
    outputs = decoder_out(h_p_1)

    # Losses
    def KL_loss(y_true, y_pred):
        return 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=-1)

    def recon_loss(y_true, y_pred):
        # return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        return K.sum(K.square(y_pred - y_true), axis=-1)

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """


        # E[log P(X|z,y)]
        #recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=1)
        recon = recon_loss(y_true= y_true,y_pred= y_pred)

        # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = KL_loss(y_true = y_true, y_pred = y_pred)

        return recon + kl


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

    cvae.compile(optimizer=optim, loss=vae_loss, metrics=[KL_loss, recon_loss])

    if verbose:
        cvae.summary()

    return(cvae, encoder, decoder)

def run_cvae(cvae, encoder, decoder, data, batchsize, nb_epochs,name, path_out):

    # Creating folder for saving
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

    # getting data
    (x_train ,y_train), (x_test, y_test) = data


    # Compile and fit
    cvae_hist = cvae.fit([x_train, y_train], x_train, batch_size=batchsize, epochs=nb_epochs,
                         validation_data=([x_test, y_test], x_test),
                         callbacks=callbacks, verbose=True)

    # Save last model
    cvae.save(os.path.join(newpath, name +  '-last_model.hdf5'))
    encoder.save(os.path.join(newpath, name +  '-last_encoder.hdf5'))
    decoder.save(os.path.join(newpath, name +  '-last_decoder.hdf5'))

    # Save history
    with open(os.path.join(newpath, name + '-history.json'), 'w') as f:
        json.dump(cvae_hist.history, f)

    plot_loss(cvae_hist, newpath)

    return cvae_hist.history


def plot_loss(cvae_hist, newpath):

    plt.plot(cvae_hist.epoch, cvae_hist.history['val_loss'])
    plt.plot(cvae_hist.epoch, cvae_hist.history['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss evolution')
    plt.legend(('test','train'))

    #plt.show()
    plt.savefig(os.path.join(newpath,'loss_evolution.png'))