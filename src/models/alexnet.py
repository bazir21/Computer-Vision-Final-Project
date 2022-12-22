# Import necessary packages
import argparse
import os
import numpy as np

# Import necessary components to build LeNet
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MEMORY_LIMIT = 2048


# https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
def alexnet_model(img_shape=(227, 227, 3), n_classes=10, l2_reg=0., weights=None):
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    # alexnet.save("alexnet.model")

    print("Bare Alexnet model created")
    return alexnet


def alexnet(x, y, retrain=False):
    x = np.array(x)
    y = np.array(y)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=MEMORY_LIMIT)])

    if os.path.isdir("alexnet.model"):
        model = keras.models.load_model("alexnet.model")
    else:
        model = alexnet_model()
        # ready training data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        # print(x_train)

        # train model
        # batch size was previous 128, but switch because of:
        # 2022-12-22 04:59:56.128030: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at matmul_op_impl.h:681 : RESOURCE_EXHAUSTED: OOM when allocating tensor with shape[262144,3072] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
        # https://github.com/tensorflow/models/issues/1993
        batch_size = 32
        epochs = 20
        print("Beginning Alexnet model training")

        # Need to compile, got the following error:
        # RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
        # So I used this: https://thecleverprogrammer.com/2021/12/13/alexnet-architecture-using-python/
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.optimizers.SGD(learning_rate=0.001),
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("alexnet.model")
        print("Finished Alexnet model training")

        model.summary()

        predictions = model.predict(x_test)
        mean_squared_error(y_test, predictions)

    # def parse_args():
    #     """
    #     Parse command line arguments.
    #     Parameters:
    #         None
    #     Returns:
    #         parser arguments
    #     """
    #     parser = argparse.ArgumentParser(description='AlexNet model')
    #     optional = parser._action_groups.pop()
    #     required = parser.add_argument_group('required arguments')
    #     optional.add_argument('--print_model',
    #                           dest='print_model',
    #                           help='Print AlexNet model',
    #                           action='store_true')
    #     parser._action_groups.append(optional)
    #     return parser.parse_args()

    return
