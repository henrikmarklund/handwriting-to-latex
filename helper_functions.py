## Visualize training history
## CODE TAKEN FROM https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
from matplotlib import pyplot as plt
import numpy as np

def visualize_training_history(fit_history):
    # list all data in history
    print(fit_history.history.keys())

    # summarize history for loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def visualize_learning_rate(learning_rates):
    # list all data in history
    #print(fit_history.history.keys())

    # summarize history for loss
    plt.plot(np.log10(learning_rates))
    plt.ylabel('Learning rate')
    plt.xlabel('epoch')
    plt.show()


def visualize_train_and_val_loss(train_loss, val_loss):
    # list all data in history

    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(val_loss)
    #plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


