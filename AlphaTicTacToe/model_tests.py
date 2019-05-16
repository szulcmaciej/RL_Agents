import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax

from nn_model import ResidualCNN


def train_model():
    model = ResidualCNN(reg_const=0,
                        learning_rate=0.1,
                        # learning_rate=0.01,
                        input_dim=(5, 5, 3),  # change for other board sizes
                        output_dim=25,  # change for other board sizes
                        hidden_layers=[{'filters': 64, 'kernel_size': 3},
                                       {'filters': 64, 'kernel_size': 3},
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
                                       ],
                        momentum=0.9)

    batch_size = 10000
    # training_states = np.zeros(shape=(batch_size, *model.input_dim))
    # training_states = np.random.randint(0, 2, size=(batch_size, *model.input_dim))
    training_states = np.random.randint(0, 2, size=(batch_size, model.input_dim[0], model.input_dim[1], model.input_dim[2]))
    # training_states[:,2,:,:] = np.zeros(shape=(training_states.shape[0], *(training_states.shape[2:])))
    training_states[:,2,:,:] = np.zeros(shape=(training_states.shape[0], training_states.shape[2], training_states.shape[3]))

    # values = np.zeros(shape=(batch_size,))
    # values = np.random.rand(batch_size)
    values = np.mean(training_states, axis=(1,2,3))

    # pis = np.zeros(shape=(batch_size, model.output_dim))
    # pis = np.random.randint(0, 2, size=(batch_size, model.output_dim))
    pis = np.mean(training_states.reshape(batch_size, 3, model.input_dim[0] ** 2), axis=1)
    # pis = np.sum(training_states.reshape(batch_size, 3, 25), axis=1)
    # pis = softmax(pis)

    pis = np.array([x / np.sum(x) for x in pis])
    # pis = np.zeros(shape=pis.shape)


    print(pis)
    print(model.model.summary())

    training_targets = [values, pis]

    fit = model.model.fit(training_states, training_targets, epochs=50, verbose=2, validation_split=0, batch_size=256)

    pred = model.model.predict(training_states[:1])
    print('pred')
    print('value')
    print(pred[0][0])
    print('policy')
    print(pred[1][0])
    print('targets')
    print('value')
    print(training_targets[0][0])
    print('policy')
    print(training_targets[1][0])

    # plt.plot(fit.history['loss'])
    # plt.plot(fit.history['value_head_loss'])
    # plt.plot(fit.history['policy_head_loss'])

    plot_history(fit.history)

    return fit


def plot_history(history, plot_from=0, plot_to=0):
    if plot_to:
        plt.plot(history['loss'][plot_from:plot_to], marker='.', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=3,
                 label='loss')
        plt.plot(history['value_head_loss'][plot_from:plot_to], marker='.', markerfacecolor='yellow', markersize=12,
                 color='lightyellow', linewidth=3, label='value_head_loss')
        plt.plot(history['policy_head_loss'][plot_from:plot_to], marker='.', markerfacecolor='green', markersize=12,
                 color='lightgreen', linewidth=3, label='policy_head_loss')
    else:
        plt.plot(history['loss'][plot_from:], marker='.', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=3,
                 label='loss')
        plt.plot(history['value_head_loss'][plot_from:], marker='.', markerfacecolor='orange', markersize=12,
                 color='lightyellow', linewidth=3, label='value_head_loss')
        plt.plot(history['policy_head_loss'][plot_from:], marker='.', markerfacecolor='green', markersize=12,
                 color='lightgreen', linewidth=3, label='policy_head_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    fit = train_model()
