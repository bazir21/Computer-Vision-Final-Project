# Import necessary packages
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

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

    # What does this do?
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=MEMORY_LIMIT)])
    #
    if os.path.isdir("alexnet.model") and not retrain:
        model = keras.models.load_model("alexnet.model")

    else:
        model = alexnet_model()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        batch_size = 1
        epochs = 20

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.optimizers.SGD(learning_rate=0.001),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("alexnet.model")
        print("Finished AlexNet model training")

    return model
