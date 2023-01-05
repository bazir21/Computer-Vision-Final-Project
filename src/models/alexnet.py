# Import necessary packages
import os
import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MEMORY_LIMIT = 2048


# New implementation follows:
# https://medium.com/analytics-vidhya/multi-class-image-classification-using-alexnet-deep-learning-network-implemented-in-keras-api-c9ae7bc4c05f

def alexnet_model(img_shape=(227, 227, 3), n_classes=10, l2_reg=0., weights=None):

    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=96, input_shape=img_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Layer 3
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # Layer 4
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Layer 5
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # Layer 6
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # Layer 7
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # Layer 8
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Layer 9.0
    model.add(Flatten())

    # Layer 9.1
    model.add(Dense(4096, input_shape=(227 * 227 * 3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Layer 10
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Layer 11
    model.add(Dense(1000))
    model.add(Activation('softmax'))

    model.summary()

    print("Bare AlexNet model created")

    return model


def alexnet(x, y, retrain=True):
    x = np.array(x)
    y = np.array(y)

    model = Sequential()
    # What does this do?
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                                                         [tf.config.experimental.VirtualDeviceConfiguration(
    #                                                             memory_limit=MEMORY_LIMIT)])
    #
    from datetime import datetime
    current_date_and_time = datetime.now()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    if os.path.isdir("alexnet.model") and retrain:
        print("Loading model.")
        model = keras.models.load_model("alexnet.model")
        model.summary()
    else:
        model = alexnet_model()

        batch_size = 10
        epochs = 100

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.optimizers.SGD(learning_rate=0.001),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("alexnet.model")
    print("Finished AlexNet model training")

    print("Generating confusion matrix:")
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions,axis=1)

    after_time = datetime.now()
    time_difference = after_time - current_date_and_time
    print(time_difference)
    f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column

    # Assigning the first subplot to graph training loss and validation loss
    ax[0].plot(history.history['loss'], color='b', label='Training Loss')
    ax[0].plot(history.history['val_loss'], color='r', label='Validation Loss')

    # Plotting the training accuracy and validation accuracy
    ax[1].plot(history.history['accuracy'], color='b', label='Training  Accuracy')
    ax[1].plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')

    plt.legend()

    def plot_confusion_matrix(figurename, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # Print Confusion matrix
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
                xticklabels = classes, yticklabels = classes,
        title = title,
        ylabel = 'True label',
        xlabel = 'Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(figurename + '.png')
        return ax

    np.set_printoptions(precision=2)

    confusion_mtx = confusion_matrix(y_test, predictions)

    class_names = ["Unripe", "Partially Ripe", "Ripe"]

    # Plotting non-normalized confusion matrix
    plot_confusion_matrix("Confusion Matrix", y_test, predictions, classes=class_names,  title='Confusion matrix, without normalization')
    plot_confusion_matrix("Confusion Matrix - Normalised", y_test, predictions, classes=class_names,  normalize=True, title='Normalized confusion matrix')

    return model
