import os
import json
from matplotlib import pyplot as plt
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras import backend as K


class BaseModel():
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.trainers = {}
        self.history = None

    def save_model(self, out_dir):
        folder = os.path.join(out_dir)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

    def main_train(self, dataset, training_epochs=100, batch_size=100, callbacks=[], verbose=True):

        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'models')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        if 'test' in dataset.keys():
            validation_data = (dataset['test']['x'], dataset['test']['y'])
        else:
            validation_data = None

        print('\n\n--- START TRAINING ---\n')
        history = self.train(dataset['train'],training_epochs, batch_size, callbacks, validation_data, verbose)

        self.history = history.history
        self.save_model(wgt_out_dir)
        self.plot_loss(res_out_dir)

        with open(os.path.join(res_out_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)

    def plot_loss(self, path_save = None):

        nb_epoch = len(self.history['loss'])

        if 'val_loss' in self.history.keys():
            best_iter = np.argmin(self.history['val_loss'])
            min_val_loss = self.history['val_loss'][best_iter]

            plt.plot(range(nb_epoch), self.history['val_loss'], label='test (min: {:0.2f}, epch: {:0.2f})'.format(min_val_loss, best_iter))

        plt.plot(range(nb_epoch), self.history['loss'], label = 'train')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss evolution')
        plt.legend()

        if path_save is not None:
            plt.savefig(os.path.join(path_save, 'loss_evolution.png'))

    #abstractmethod
    def train(self, training_dataset,training_epochs, batch_size, callbacks, validation_data, verbose):
        '''
        Plase override "train" method in the derived model!
        '''

        pass


class CVAE(BaseModel):
    def __init__(self, input_dim=96, cond_dim=12, z_dim=2, e_dims=[24], d_dims=[24], beta=1, verbose=True,**kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.e_dims = e_dims
        self.d_dims = d_dims
        self.beta = beta
        self.encoder = None
        self.decoder = None
        self.cvae = None
        self.verbose = verbose

        self.build_model()

    def build_model(self):
        """

        :param verbose:
        :return:
        """

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        x_true = Input(shape=(self.input_dim,), name='x_true')
        cond_true = Input(shape=(self.cond_dim,), name='cond')

        # Encoding
        z_mu, z_log_sigma = self.encoder([x_true, cond_true])

        # Sampling
        def sample_z(args):
            mu, log_sigma = args
            eps = K.random_normal(shape=(K.shape(mu)[0], self.z_dim), mean=0., stddev=1.)
            return mu + K.exp(log_sigma / 2) * eps

        z = Lambda(sample_z, name='sample_z')([z_mu, z_log_sigma])

        # Decoding
        x_hat = self.decoder([z, cond_true])

        # Defining loss
        vae_loss, recon_loss, kl_loss = self.build_loss(z_mu, z_log_sigma)

        # Defining and compiling cvae model
        self.cvae = Model(inputs=[x_true, cond_true], outputs=x_hat)
        self.cvae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[kl_loss, recon_loss])

        # Store trainers
        self.store_to_save('cvae')
        self.store_to_save('encoder')
        self.store_to_save('decoder')

        if self.verbose:
            print("complete model: ")
            self.cvae.summary()
            print("encoder: ")
            self.encoder.summary()
            print("decoder: ")
            self.decoder.summary()

    def build_encoder(self):
        """
        Encoder: Q(z|X,y)
        :return:
        """

        x_inputs = Input(shape=(self.input_dim,), name='enc_x_true')
        cond_inputs = Input(shape=(self.cond_dim,), name='enc_cond')
        x = concatenate([x_inputs, cond_inputs], name='enc_input')

        for idx, layer_dim in enumerate(self.e_dims):
            x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)

        z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
        z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)

        return Model(inputs=[x_inputs, cond_inputs], outputs=[z_mu, z_log_sigma], name='encoder')

    def build_decoder(self):
        """
        Decoder: P(X|z,y)
        :return:
        """

        x_inputs = Input(shape=(self.z_dim,), name='dec_z')
        cond_inputs = Input(shape=(self.cond_dim,), name='dec_cond')
        x = concatenate([x_inputs, cond_inputs], name='dec_input')

        for idx, layer_dim in enumerate(self.d_dims):
            x = Dense(units=layer_dim, activation='relu', name='dec_dense_{}'.format(idx))(x)

        output = Dense(units=self.input_dim, activation='linear', name='dec_x_hat')(x)

        return Model(inputs=[x_inputs, cond_inputs], outputs=output, name='decoder')

    def build_loss(self, z_mu, z_log_sigma):
        """

        :return:
        """

        def kl_loss(y_true, y_pred):
            return 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mu) - 1. - z_log_sigma, axis=-1)

        def recon_loss(y_true, y_pred):
            # return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
            return K.sum(K.square(y_pred - y_true), axis=-1)

        def vae_loss(y_true, y_pred):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

            # E[log P(X|z,y)]
            recon = recon_loss(y_true=y_true, y_pred=y_pred)

            # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = kl_loss(y_true=y_true, y_pred=y_pred)

            return recon + self.beta*kl

        return vae_loss, recon_loss, kl_loss

    def train(self, dataset_train, training_epochs=10, batch_size=20, callbacks = [], validation_data = None, verbose = True):
        """

        :param dataset_train:
        :param training_epochs:
        :param batch_size:
        :param callbacks:
        :param validation_data:
        :param verbose:
        :return:
        """

        assert len(dataset_train) >= 2  # Check that both x and cond are present

        cvae_hist = self.cvae.fit(dataset_train['x'], dataset_train['y'], batch_size=batch_size, epochs=training_epochs,
                             validation_data=validation_data,
                             callbacks=callbacks, verbose=True)

        return cvae_hist


class CVAE_temp(CVAE):
    """
    Improvement of CVAE that encode the temperature as a condition
    """
    def __init__(self, to_emb_dim=96, cond_pre_dim=12, emb_dims=[2], **kwargs):

        self.to_emb_dim = to_emb_dim
        self.cond_pre_dim = cond_pre_dim
        self.emb_dims = emb_dims
        self.embedding = None

        super().__init__(cond_dim=self.cond_pre_dim + self.emb_dims[-1],**kwargs)

    def build_model(self):
        """

        :param verbose:
        :return:
        """

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.embedding = self.build_embedding()

        x_true = Input(shape=(self.input_dim,), name='x_true')
        to_emb = Input(shape=(self.to_emb_dim,), name='to_emb')
        cond_pre = Input(shape=(self.cond_pre_dim,), name='cond_pre')

        cond_emb = self.embedding(inputs=to_emb)

        cond_true = concatenate([cond_pre, cond_emb], name='conc_cond')

        # Encoding
        z_mu, z_log_sigma = self.encoder([x_true, cond_true])

        # Sampling
        def sample_z(args):
            mu, log_sigma = args
            eps = K.random_normal(shape=(K.shape(mu)[0], self.z_dim), mean=0., stddev=1.)
            return mu + K.exp(log_sigma / 2) * eps

        z = Lambda(sample_z, name='sample_z')([z_mu, z_log_sigma])

        # Decoding
        x_hat = self.decoder([z, cond_true])

        # Defining loss
        vae_loss, recon_loss, kl_loss = self.build_loss(z_mu, z_log_sigma)

        # Defining and compiling cvae model
        self.cvae = Model(inputs=[x_true, cond_pre, to_emb], outputs=x_hat)
        self.cvae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[kl_loss, recon_loss])

        # Store trainers
        self.store_to_save('cvae')

        self.store_to_save('encoder')
        self.store_to_save('decoder')

        if self.verbose:
            print("complete model: ")
            self.cvae.summary()
            print("embedding: ")
            self.embedding.summary()
            print("encoder: ")
            self.encoder.summary()
            print("decoder: ")
            self.decoder.summary()


    def build_embedding(self):
        """
        Embedding of the temperature
        :return:
        """

        x_inputs = Input(shape=(self.to_emb_dim,), name='emb_input')

        x = x_inputs

        for idx, layer_dim in enumerate(self.emb_dims[:-1]):
            x = Dense(units=layer_dim, activation='relu', name="emb_dense_{}".format(idx))(x)

        embedding = Dense(units=self.emb_dims[-1], activation='linear', name="emb_dense_last")(x)

        return Model(inputs=x_inputs, outputs=embedding, name='embedding')


    def train(self, dataset_train, training_epochs=10, batch_size=20, callbacks = [], validation_data = None, verbose = True):
        """

        :param dataset_train:
        :param training_epochs:
        :param batch_size:
        :param callbacks:
        :param validation_data:
        :param verbose:
        :return:
        """

        assert len(dataset_train) >= 3  # Check that both x cond and temp are presents

        cvae_hist = self.cvae.fit(dataset_train['x'], dataset_train['y'], batch_size=batch_size, epochs=training_epochs,
                             validation_data=validation_data,
                             callbacks=callbacks, verbose=True)

        return cvae_hist