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


__all__ = ['Propagator_serial', 'get_mixture_loss_func_serial']


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

        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus', activation='sigmoid')  # mix*output vals, no activation
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = Dense(self.num_mix, name='mdn_pi', activation='softmax')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)
        

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        self._trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        self._non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        super(MDN, self).build(input_shape)


    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x),
                                                self.mdn_sigmas(x),
                                                self.mdn_pi(x)],
                                               name='mdn_outputs')
        return mdn_out


    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)


    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_mixture_loss_func_serial(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        
        mus = tf.reshape(out_mu, (-1, num_mixes, output_dim))
        sigs = tf.reshape(out_sigma, (-1, num_mixes, output_dim))
        
        cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
        
        mixture = tfd.MixtureSameFamily(
            mixture_distribution=cat, 
            components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs) 
        )
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss_func
    with tf.name_scope('MDN'):
        return loss_func


def get_mixture_fun_serial(output_dim, num_mixes):
    """Construct a TensorFlor sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    def sampling_func(y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
        

        mus = tf.reshape(out_mu, (-1, num_mixes, output_dim))
        sigs = tf.reshape(out_sigma, (-1, num_mixes, output_dim))
        
        cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
        
        mixture = tfd.MixtureSameFamily(
            mixture_distribution=cat, 
            components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs) 
        )        

        return mixture

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func


class Propagator_serial(BaseEstimator, TransformerMixin):

    def __init__(self, xp_idxs, xt_idxs, n_components=2, lag_time=100, batch_size=1000, 
                learning_rate=0.001, n_epochs=100, hidden_layer_depth=2, 
                hidden_size=100, activation='swish', callbacks=None, verbose=True):
        
        # xp is what you only know at previous times and want to predict
        # xt is what you know at current time and use as input only
        
        self.xp_idxs = np.array(xp_idxs)
        self.xt_idxs = np.array(xt_idxs)
        self.xp_dim = len(xp_idxs) 
        self.xt_dim = len(xt_idxs) 
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lag_time = lag_time
        self.callbacks = callbacks
        self.n_components = n_components

        model = keras.Sequential()
        get_custom_objects().update({'swish': swish})
        
        model.add(Dense(hidden_size, activation=activation, input_shape=(self.xp_dim + self.xt_dim,)))

        for _ in range(hidden_layer_depth - 1):
            model.add(Dense(hidden_size, activation=activation))
        
        model.add(MDN(self.xp_dim, n_components))
        model.compile(loss=get_mixture_loss_func_serial(self.xp_dim, n_components), 
                      optimizer=keras.optimizers.Adam(lr=learning_rate))

        self.model = model
        self.is_fitted = False
        self.sess = K.get_session()

    
    def fit(self, X, y=None): 
        x0, xt = self._create_dataset(X)        
        print(x0.shape, xt.shape)
        
        self.model.fit(x0, xt, batch_size=self.batch_size, 
                    epochs=self.n_epochs, verbose=self.verbose,
                    callbacks=self.callbacks)
        
        self.is_fitted = True
        return self

    
    def transform(self, X):

        if self.is_fitted:
            out = self.model.predict(X, batch_size=self.batch_size)
            return out
        
        raise RuntimeError('Model needs to be fit first.')


    def _create_dataset(self, data, lag_time=None):
        '''Inputs are all xp_idxs at x0 and xt_idxs at xt
           Outputs are only xp_idxs at xt
            '''
 
        if lag_time is None:
            lag_time = self.lag_time

        if type(data) is list: 
            x_t0 = []
            x_tt = []

            for item in data:
                
                xp_t0 = item[:-lag_time, self.xp_idxs]
                xp_tt = item[lag_time:, self.xp_idxs]
                
                # when passing in tt data
                if len(self.xt_idxs) > 0:
                    xt_tt = item[lag_time:, self.xt_idxs]
                    x_t0.append(np.concatenate([xt_tt, xp_t0], axis=1))
                    
                # account for standard prop case:
                else:
                    x_t0.append(xp_t0)
                x_tt.append(xp_tt)
            
            x_t0 = np.concatenate(x_t0)
            x_tt = np.concatenate(x_tt) 
            
        elif type(data) is np.ndarray:
            
            xp_t0 = item[:-lag_time, self.xp_idxs]
            xp_tt = item[lag_time:, self.xp_idxs]
        
            if len(self.xt_idxs) > 0:
                xt_tt = item[lag_time:, self.xt_idxs]
                x_t0 = (np.concatenate([xt_tt, xp_t0], axis=1))
            else:
                x_t0 = xp_t0
            x_tt = xp_tt
            
        else:
            raise TypeError('Data type {} is not supported'.format(type(data)))

        return [x_t0, x_tt]

    def propagate(self, x0, xt=None, n_steps=1):
        '''Propagation proceeds as a function of x0 x_tp and x1 x_tt
           x0:  1 x N_p dim corresponding to initial state of predicted quantities
           xt:  t x N_t dim of previously propagated coords
           '''
      
        mix_fun = get_mixture_fun_serial(self.xp_dim, self.n_components)
        
        # append first xp_1 to x0 for initial state
        if xt is None:
            xt = []
        else:
            xt_1 = tf.expand_dims(xt[1], axis=0)
            x0 = tf.concat([xt_1, x0], axis=1)
            
        
        def body(loop_counter, x, xt, x_out):
            y = self.model(x)
          
            # sample new coords from MDN
            mixture = mix_fun(y)
            xp = mixture.sample()
            
            # only runs if xt data is provided
            if not isinstance(xt, list):
                
                # gets current time component of previous info
                xt_t = tf.expand_dims(xt[loop_counter], axis=0)

                # concatenate known value for next round input
                xp = tf.concat([xt_t, xp], axis=1)
                
            # periodic wrap any values outside (0, 1) -- add flag for this? Use standard clip for now
            #xp = tf.math.floormod(xp, tf.ones(xp.shape))
            xp = tf.clip_by_value(xp, 0, 1)
            
            # pass output into next round input
            x_out = x_out.write(loop_counter, xp)
            
            return [loop_counter + 1, xp, xt, x_out]
    
        def cond(loop_counter, x, xt, x_out):
            return tf.less(loop_counter, n_steps)
        
        with tf.variable_scope('propagate', reuse=tf.AUTO_REUSE):
            x_out = tf.TensorArray(size=n_steps, dtype=tf.float32)
            loop_counter = tf.constant(0)

            result = tf.while_loop(cond, body, [loop_counter, x0, xt, x_out])
            traj = result[-1].stack()
            coords = self.sess.run(traj)
            
            return coords[:, :, self.xt_dim:] #coords 
