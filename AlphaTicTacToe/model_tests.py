import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax

from AlphaTicTacToe.nn_model import ResidualCNN


def train_model():
    model = ResidualCNN(reg_const=0,
                        learning_rate=0.01,
                        input_dim=(5, 5, 3),  # change for other board sizes
                        output_dim=25,  # change for other board sizes
                        hidden_layers=[{'filters': 64, 'kernel_size': 3},
                                       {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       # {'filters': 64, 'kernel_size': 3},
                                       {'filters': 64, 'kernel_size': 3}, ],
                        momentum=0.9)

    batch_size = 1000
    # training_states = np.zeros(shape=(batch_size, *model.input_dim))
    training_states = np.random.randint(0, 2, size=(batch_size, *model.input_dim))
    training_states[:,2,:,:] = np.zeros(shape=(training_states.shape[0], *(training_states.shape[2:])))

    # values = np.zeros(shape=(batch_size,))
    # values = np.random.rand(batch_size)
    values = np.mean(training_states, axis=(1,2,3))

    # pis = np.zeros(shape=(batch_size, model.output_dim))
    # pis = np.random.randint(0, 2, size=(batch_size, model.output_dim))
    pis = np.mean(training_states.reshape(batch_size, 3, 25), axis=1)
    # pis = np.sum(training_states.reshape(batch_size, 3, 25), axis=1)
    # pis = softmax(pis)

    pis = np.array([x / np.sum(x) for x in pis])
    # pis = np.zeros(shape=pis.shape)


    print(pis)
    print(model.model.summary())

    training_targets = [values, pis]

    fit = model.model.fit(training_states, training_targets, epochs=40, verbose=2, validation_split=0, batch_size=64)

    # model.model.load_weights('../saved_models/model_weights.h5')

    pred = model.model.predict(training_states[:1])
    print('value')
    print('real')
    print(training_targets[0][0])
    print('pred')
    print(pred[0])
    print()

    print('policy')
    print('real')
    print(training_targets[1][0])
    print('pred')
    print(pred[1])

    score = model.model.evaluate(training_states[:1], [training_targets[0][:1], training_targets[1][:1]])
    print(score)

    plt.plot('loss', data=fit.history, marker='.', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=3, label='loss')
    plt.plot('value_head_loss', data=fit.history, marker='.', markerfacecolor='yellow', markersize=12, color='lightyellow', linewidth=3, label='value_head_loss')
    plt.plot('policy_head_loss', data=fit.history, marker='.', markerfacecolor='green', markersize=12, color='lightgreen', linewidth=3, label='policy_head_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

    model.model.save_weights('../saved_models/model_weights.h5')


if __name__ == '__main__':
    train_model()
