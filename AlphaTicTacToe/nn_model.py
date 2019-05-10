import logging
# import config
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
import tensorflow as tf

# from loss import softmax_cross_entropy_with_logits

# import loggers as lg

import keras.backend as K


# from settings import run_folder, run_archive_folder


def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=p)

    return loss


class ResidualCNN:
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers, momentum):
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            # , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return x

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            # , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def value_head(self, x):

        x = Conv2D(
            filters=1
            , kernel_size=(1, 1)
            # , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            20
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1
            , use_bias=False
            , activation='tanh'
            , kernel_regularizer=regularizers.l2(self.reg_const)
            , name='value_head'
        )(x)

        return x

    def policy_head(self, x):

        x = Conv2D(
            filters=2
            , kernel_size=(1, 1)
            # , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim
            , use_bias=False
            # , activation='linear'
            , activation='softmax'
            , kernel_regularizer=regularizers.l2(self.reg_const)
            , name='policy_head'
        )(x)

        return x

    def _build_model(self):

        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
                      loss_weights={'value_head': 1, 'policy_head': 1}
                      )

        return model

    # def convertToModelInput(self, state):
    #     inputToModel = state.binary  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
    #     inputToModel = np.reshape(inputToModel, self.input_dim)
    #     return (inputToModel)