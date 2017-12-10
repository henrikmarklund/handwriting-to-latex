## Visualize training history
## CODE TAKEN FROM https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
from matplotlib import pyplot as plt

def visualize_training_history(fit_history):
    # list all data in history
    print(fit_history.history.keys())

    # summarize history for loss
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()