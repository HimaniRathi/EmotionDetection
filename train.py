import getopt
import sys
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Conv3D, MaxPooling3D, \
    ZeroPadding3D, GRU, TimeDistributed

import numpy as np

datadir = "data/"
model = Sequential()
input_shape = (48, 48, 1)
output_shape = 7


def save_model(model, index=""):
    model_json = model.to_json()
    with open(datadir + "model" + index + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(datadir + "model" + index + ".h5")


def cnn_image_based():
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    conv_arch = [(32, 2), (64, 2), (128, 2)]
    dense = [64, 2]
    if (conv_arch[0][1] - 1) != 0:
        for i in range(conv_arch[0][1] - 1):
            model.add(Conv2D(conv_arch[0][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            model.add(Conv2D(conv_arch[1][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            model.add(Conv2D(conv_arch[2][0], kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # if conv_arch[3][1] != 0:
    #     for i in range(conv_arch[3][1]):
    #         model.add(Conv2D(conv_arch[3][0], kernel_size=(3, 3), padding='same', activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    if dense[1] != 0:
        for i in range(dense[1]):
            model.add(Dense(dense[0], activation='relu'))
            model.add(Dropout(0.25))
    model.add(Dense(output_shape, activation='softmax'))
    # 16 layers


def cnn_lstm():
    model.add(LSTM(256, return_sequences=False, dropout=0.5))

    model.add(Dense(output_shape, activation='softmax'))


def model2cnn_lstm():
    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape[1:]))
    conv_arch = [(32, 2), (64, 2), (128, 2)]
    dense = [64, 2]
    if (conv_arch[0][1] - 1) != 0:
        for i in range(conv_arch[0][1] - 1):
            cnn.add(Conv2D(conv_arch[0][0], kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[1][1] != 0:
        for i in range(conv_arch[1][1]):
            cnn.add(Conv2D(conv_arch[1][0], kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

    if conv_arch[2][1] != 0:
        for i in range(conv_arch[2][1]):
            cnn.add(Conv2D(conv_arch[2][0], kernel_size=(3, 3), padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())  # this converts 3D feature maps to 1D feature vectors
    # if dense[1] != 0:
    #     for i in range(dense[1]):
    #         cnn.add(Dense(dense[0], activation='relu'))
    #         cnn.add(Dropout(0.25))
    # cnn.add(Dense(output_shape, activation='softmax'))
    # 16 layers

    model.add(TimeDistributed(cnn, input_shape=input_shape))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(output_shape, activation='softmax'))


def c3d():
    model.add(Conv3D(64, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv1',
                     subsample=(1, 1, 1),
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv2',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv4a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv4b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))

    # 5th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv5a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv5b',
                     subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))


def lstm():
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(60, 4608)))
    model.add(Dense(output_shape, activation='softmax'))


def main(argv):
    global opts
    global datadir
    global input_shape
    help_text = """train.py  -h --help
    -h --help       Get help.
    -t --test       Test data dir
    -m --model      Test mode.(place first)
    """
    try:
        opts, args = getopt.getopt(argv, "htm:")
    except getopt.GetoptError:
        print(help_text)
        sys.exit()
    x_fname = datadir + 'x_train.npy'
    y_fname = datadir + 'y_train.npy'
    x_train = np.load(x_fname)
    y_train = np.load(y_fname)
    print('Loading data...')
    print(x_train.shape, y_train.shape)
    model_number = 0
    for opt, arg in opts:
        if opt in ('-t', '--test'):
            datadir = "datat/"
        if opt in ('-h', '--help'):
            print(help_text)
            sys.exit()
        elif opt in ('-m', '--model'):
            model_number = int(arg)

    # apply required model
    if model_number == 0:
        # loading image data
        input_shape = (48, 48, 1)
        x_train = np.load(datadir + 'x_train_image.npy')
        y_train = np.load(datadir + 'y_train_image.npy')
        print('Loading image data...')
        print(x_train.shape, y_train.shape)
        cnn_image_based()
        print("CNN Image based")
    elif model_number == 1:
        # loading image data

        x_train = np.load(datadir + 'x_train_vec.npy')
        # y_train = np.load(datadir + 'y_train.npy')
        print('Loading vector data...')
        print(x_train.shape, y_train.shape)
        lstm()
    elif model_number == 2:
        c3d()
    elif model_number == 3:
        input_shape = (60, 48, 48, 1)
        model2cnn_lstm()
    elif model_number == 4:
        pass

    # compiling
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print('Training....')
    print(x_train.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    hist = model.fit(x=x_train, y=y_train, epochs=10, batch_size=10, callbacks=[early_stopping],
                     validation_split=0.2, shuffle=True, verbose=1)
    train_val_accuracy = hist.history

    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])
    save_model(model, index="imagebased")
    print("Saved model to disk")


if __name__ == "__main__":
    # sys.stdout = open('logs/training0.txt', 'w')
    # sys.stderr = open('logs/training0-err.txt', 'w')
    main(sys.argv[1:])
