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


run_folder = './run/'
run_archive_folder = './run_archive/'


def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi, logits=p)

    return loss


class GenModel:
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)

    def write(self, game, version):
        self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

    def read(self, game, run_number, version):
        return load_model(
            run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(
                version) + '.h5',
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    # def printWeightAverages(self):
    #     layers = self.model.layers
    #     for i, l in enumerate(layers):
    #         try:
    #             x = l.get_weights()[0]
    #             lg.logger_model.info('WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i,
    #                                  np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
    #         except:
    #             pass
    #     lg.logger_model.info('------------------')
    #     for i, l in enumerate(layers):
    #         try:
    #             x = l.get_weights()[1]
    #             lg.logger_model.info('BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)),
    #                                  np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
    #         except:
    #             pass
    #     lg.logger_model.info('******************')

    # def viewLayers(self):
    #     layers = self.model.layers
    #     for i, l in enumerate(layers):
    #         x = l.get_weights()
    #         print('LAYER ' + str(i))
    #
    #         try:
    #             weights = x[0]
    #             s = weights.shape
    #
    #             fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
    #             channel = 0
    #             filter = 0
    #             for i in range(s[2] * s[3]):
    #                 sub = fig.add_subplot(s[3], s[2], i + 1)
    #                 sub.imshow(weights[:, :, channel, filter], cmap='coolwarm', clim=(-1, 1), aspect="auto")
    #                 channel = (channel + 1) % s[2]
    #                 filter = (filter + 1) % s[3]
    #
    #         except:
    #
    #             try:
    #                 fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
    #                 for i in range(len(x)):
    #                     sub = fig.add_subplot(len(x), 1, i + 1)
    #                     if i == 0:
    #                         clim = (0, 2)
    #                     else:
    #                         clim = (0, 2)
    #                     sub.imshow([x[i]], cmap='coolwarm', clim=clim, aspect="auto")
    #
    #                 plt.show()
    #
    #             except:
    #                 try:
    #                     fig = plt.figure(figsize=(3, 3))  # width, height in inches
    #                     sub = fig.add_subplot(1, 1, 1)
    #                     sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")
    #
    #                     plt.show()
    #
    #                 except:
    #                     pass
    #
    #         plt.show()
    #
    #     lg.logger_model.info('------------------')


class ResidualCNN(GenModel):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers, momentum):
        GenModel.__init__(self, reg_const, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        # self.reg_const = reg_const
        # self.learning_rate = learning_rate
        self.momentum = momentum
        # self.input_dim = input_dim
        # self.output_dim = output_dim
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
            , activation='linear'
            # , activation='softmax'
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
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'mean_squared_error'},
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
                      loss_weights={'value_head': 1, 'policy_head': 1}
                      )

        return model

    def convertToModelInput(self, state):
        numpy_board = state.board
        nn_board = np.zeros(shape=(3, numpy_board.shape[0], numpy_board.shape[1]))
        nn_board[0, :, :] = (numpy_board == 1)
        nn_board[1, :, :] = (numpy_board == 2)
        nn_board[2, :, :] = state.playerTurn - 1

        return nn_board

    # def _build_model(self):
    #
    #     main_input = Input(shape=self.input_dim, name='main_input')
    #
    #     x = Flatten()(main_input)
    #
    #     #HIDDEN
    #
    #     x = Dense(
    #         128
    #         , use_bias=True
    #         , activation='relu'
    #         , kernel_regularizer=regularizers.l2(self.reg_const)
    #     )(x)
    #
    #     x = Dense(
    #         128
    #         , use_bias=True
    #         , activation='relu'
    #         , kernel_regularizer=regularizers.l2(self.reg_const)
    #     )(x)
    #
    #
    #
    #     #HEADS
    #
    #     vh = Dense(
    #         20
    #         , use_bias=False
    #         , activation='linear'
    #         , kernel_regularizer=regularizers.l2(self.reg_const)
    #     )(x)
    #
    #     vh = LeakyReLU()(vh)
    #
    #     vh = Dense(
    #         1
    #         , use_bias=False
    #         , activation='tanh'
    #         , kernel_regularizer=regularizers.l2(self.reg_const)
    #         , name='value_head'
    #     )(vh)
    #
    #
    #     ph = Dense(
    #         self.output_dim
    #         , use_bias=False
    #         , activation='softmax'
    #         , kernel_regularizer=regularizers.l2(self.reg_const)
    #         , name='policy_head'
    #     )(x)
    #
    #     model = Model(inputs=[main_input], outputs=[vh, ph])
    #     # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
    #     # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
    #     model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'mean_absolute_error'},
    #                   optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
    #                   loss_weights={'value_head': 1, 'policy_head': 1}
    #                   )
    #
    #     return model

    # def convertToModelInput(self, state):
    #     inputToModel = state.binary  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
    #     inputToModel = np.reshape(inputToModel, self.input_dim)
    #     return (inputToModel)