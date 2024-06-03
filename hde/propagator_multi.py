import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects, plot_model
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

__all__ = ['Propagator']


def swish(x):
    return (K.sigmoid(x) * x)


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return K.elu(x) + 1 + K.epsilon()


class MDN(Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        
        print('\n', 'init layer', self.output_dim, self.num_mix, '\n')

        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus' ,activation='sigmoid') 
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')
            self.mdn_pi = Dense(self.num_mix*3, name='mdn_pi', activation='softmax')  # mix vals, logit
            
        super(MDN, self).__init__(**kwargs)
        

    def build(self, input_shape):
        
        # this is the hidden layer size
        print('input shape: ', input_shape)
        
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        self._trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        
        print('\nTrainable weights -- should add up to n_hidden x 270 (for n_mix=10):')
        print(self.mdn_mus.trainable_weights)
        print(self.mdn_sigmas.trainable_weights)
        print(self.mdn_pi.trainable_weights)
        
        self._non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        
        super(MDN, self).build(input_shape)
        

    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x),
                                                self.mdn_sigmas(x),
                                                self.mdn_pi(x)],
                                               name='mdn_outputs')
            print('\nmdn_out', mdn_out)
            
        return mdn_out


    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        
        #return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)
        
        # I don't think this function is actually doing anything
        # the loss runs by stealing values form the batch dimension, sneaky...
        
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + (3 * self.num_mix))


    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_multi_mixture_loss_func(output_dim, num_mixes, idxs_list):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    
    def loss_func(y_true, y_pred):
        '''Create distinct mixture models '''

        srv_idxs = [0, 1, 2, 3, 4, 5]
        trans_idxs = [6, 7, 8]
        angle_idxs = [9, 10, 11]
        idxs_list = [srv_idxs, trans_idxs, angle_idxs]
        print('\nstarting loss_func, idxs_list = ', idxs_list, '\n') 
        
        print('\ny_pred:', y_pred)
        
        # is line redundant, or does it handle the batch?
        if (2 * num_mixes * output_dim) + 3 * num_mixes > 260:
            y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + 3 * num_mixes], name='reshape_ypreds')
            
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        
        print('\ny_pred:', y_pred)
        
        # split y_pred and y_true for each independent model
        model_dims = [len(idxs) for idxs in idxs_list]
        model_params = [2*num_mixes*dims + num_mixes for dims in model_dims]
        print('\model params:', model_params)
        
        y_true_list = tf.split(y_true, num_or_size_splits=model_dims, axis=-1)
        y_pred_list = tf.split(y_pred, num_or_size_splits=model_params, axis=-1)
        
        print('\ny_pred_list:', y_pred_list)
        
        # instantiate loss
        loss = tf.constant(0.0)
        
        # iterate through each model
        for idxs, y_p, y_t in zip(idxs_list, y_pred_list, y_true_list):

            # Split the inputs into paramaters
            model_dim = len(idxs)
            out_mu, out_sigma, out_pi = tf.split(y_p, num_or_size_splits=[num_mixes * model_dim,
                                                                             num_mixes * model_dim,
                                                                             num_mixes],
                                                 axis=-1, name='mdn_coef_split')

            mus = tf.reshape(out_mu, (-1, num_mixes, model_dim))
            sigs = tf.reshape(out_sigma, (-1, num_mixes, model_dim))
            cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
            
            # create distributions only along select modes
            mixture = tfd.MixtureSameFamily(
                mixture_distribution=cat,
                components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs))

            # check negative log_prob along along select modes
            loss_model = mixture.log_prob(y_t)
            loss_model = tf.negative(loss_model)
            loss_model = tf.reduce_mean(loss_model)

            # add to total loss
            loss = tf.add(loss, loss_model)
        
        print('\nReturning loss\n')
       
        return loss

    # Actually return the loss_funcloss = tf.constant(0.0)
    with tf.name_scope('MDN'):
        return loss_func


def get_mixture_fun(output_dim, num_mixes, idxs_list):
    """Construct a TensorFlor sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    
    def sampling_func(y_pred):
            
        # pass in each set of idxs seperately
        srv_idxs = [0, 1, 2, 3, 4, 5]
        trans_idxs = [6, 7, 8]
        angle_idxs = [9, 10, 11]
        idxs_list = [srv_idxs, trans_idxs, angle_idxs]
        
        # can this add dimensions???
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + 3 * num_mixes], name='reshape_ypreds')
        
        # split y_pred and y_true for each independent model
        model_dims = [len(idxs) for idxs in idxs_list]
        model_params = [2*num_mixes*dims + num_mixes for dims in model_dims]
        y_pred_list = tf.split(y_pred, num_or_size_splits=model_params, axis=-1)
        
        # collect each mixture model in a list
        mixture_list = []
        
        # iterate through each model
        for idxs, y_pred in zip(idxs_list, y_pred_list):

            # Split the inputs into paramaters
            model_dim = len(idxs)
            out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * model_dim,
                                                                             num_mixes * model_dim,
                                                                             num_mixes],
                                                 axis=-1, name='mdn_coef_split')

            mus = tf.reshape(out_mu, (-1, num_mixes, model_dim))
            sigs = tf.reshape(out_sigma, (-1, num_mixes, model_dim))
            cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
            
            # create distributions only along select modes
            mixture = tfd.MixtureSameFamily(
                mixture_distribution=cat, 
                components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs))
            mixture_list.append(mixture)

        return mixture_list
    
    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func


class Propagator(BaseEstimator, TransformerMixin):

    def __init__(self, input_dim, n_components=2, lag_time=100, batch_size=1000, 
                learning_rate=0.001, n_epochs=100, hidden_layer_depth=2, 
                hidden_size=100, activation='swish', callbacks=None, verbose=True, alpha=0.0):
        
        self.input_dim = input_dim 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lag_time = lag_time
        self.callbacks = callbacks
        self.n_components = n_components
        self.alpha = alpha
        
        # pass in each set of idxs seperately
        srv_idxs = [0, 1, 2, 3, 4, 5]
        trans_idxs = [6, 7, 8]
        angle_idxs = [9, 10, 11]
        idxs_list = [srv_idxs, trans_idxs, angle_idxs]
        self.idxs_list =  idxs_list 

        model = keras.Sequential()
        get_custom_objects().update({'swish': swish})
       
        model.add(Dense(hidden_size, activation=activation, input_shape=(input_dim,)))
        for _ in range(hidden_layer_depth - 1):
            model.add(Dense(hidden_size, activation=activation))
        
        model.add(MDN(input_dim, n_components))
        model.compile(loss=get_multi_mixture_loss_func(input_dim, n_components, idxs_list), 
                      optimizer=keras.optimizers.Adam(lr=learning_rate))

        self.model = model
        print(self.model.summary())
        
        self.is_fitted = False
        self.sess = K.get_session()


    def fit(self, X, y=None): 
        x0, xt = self._create_dataset(X)
        
        print('\nstarting fit\n')
        print(self.model.summary())
        
        self.model.fit(x0, xt, batch_size=self.batch_size, 
                    epochs=self.n_epochs, verbose=self.verbose,
                    callbacks=self.callbacks)
        
        # iterate through each mo
        self.is_fitted = True
        return self

    
    def transform(self, X):

        if self.is_fitted:
            out = self.model.predict(X, batch_size=self.batch_size)
            return out
        
        raise RuntimeError('Model needs to be fit first.')


    def _create_dataset(self, data, lag_time=None):
        if lag_time is None:
            lag_time = self.lag_time

        if type(data) is list: 
            x_t0 = []
            x_tt = []
            for item in data:
                x_t0.append(item[:-lag_time])
                x_tt.append(item[lag_time:])
            
            x_t0 = np.concatenate(x_t0)
            x_tt = np.concatenate(x_tt)
            
        elif type(data) is np.ndarray:
            x_t0 = data[:-lag_time]
            x_tt = data[lag_time:]
        
        else:
            raise TypeError('Data type {} is not supported'.format(type(data)))

        return [x_t0, x_tt]

    def propagate(self, x0, n_steps=1):
        mix_fun = get_mixture_fun(self.input_dim, self.n_components, self.idxs_list)
        
        print('mix_fun', mix_fun)
        
        def old_body(loop_counter, x, x_out):
            y = self.model(x)
            mixture = mix_fun(y)   # MJ added mus output
    
            xn = mixture.sample()
            xn = tf.clip_by_value(xn, 0, 1)
            x_out = x_out.write(loop_counter, xn)
        
            return [loop_counter + 1, xn, x_out]
        
        def body(loop_counter, x, x_out):
            
            # obtain a list containing each mixture model
            y = self.model(x)
            mixture_list = mix_fun(y)
            xn = []
            
            # sample each distribution and append to list
            for mixture in mixture_list:
                xn.append(mixture.sample())
                
            # stack and clip each sample
            xn = tf.concat(xn, axis=1)
            xn = tf.clip_by_value(xn, 0, 1)
            x_out = x_out.write(loop_counter, xn)
        
            return [loop_counter + 1, xn, x_out]
    
        def cond(loop_counter, x, x_out):
            return tf.less(loop_counter, n_steps)
        
        with tf.variable_scope('propagate', reuse=tf.AUTO_REUSE):
            x_out = tf.TensorArray(size=n_steps, dtype=tf.float32)
            loop_counter = tf.constant(0)

            result = tf.while_loop(cond, body, [loop_counter, x0, x_out])
            traj = result[-1].stack()
            coords = self.sess.run(traj)
            return coords
