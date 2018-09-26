import getopt
import sys
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import numpy as np

datadir = "data/"


def save_model(model, index=""):
    model_json = model.to_json()
    with open(datadir + "model" + index + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(datadir + "model" + index + ".h5")


def cnn_image_based(x_train, y_train, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
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
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # 16 layers
    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Training....')

    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split, shuffle=True, verbose=1)
    train_val_accuracy = hist.history

    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    print('          Done!')
    print('     Train acc: ', train_acc[-1])
    print('Validation acc: ', val_acc[-1])
    print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])
    save_model(model, index="imagebased")
    print("Saved model to disk")


def main(argv):
    global opts
    global datadir
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
    for opt, arg in opts:
        if opt in ('-t', '--test'):
            datadir = "datat/"
        if opt in ('-h', '--help'):
            print(help_text)
            sys.exit()
        elif opt in ('-m', '--model'):
            model = int(arg)
            if model == 0:
                x_train = np.load(datadir+'x_train_image.npy')
                y_train = np.load(datadir+'y_train_image.npy')
                print('Loading image data...')
                print(x_train.shape, y_train.shape)
                cnn_image_based(x_train, y_train, batch_size=10, validation_split=0.2, epochs=50)
            sys.exit()
    print(help_text)


if __name__ == "__main__":
    # sys.stdout = open('logs/cleaning.txt', 'w')
    # sys.stderr = open('logs/cleaning-err.txt', 'w')
    main(sys.argv[1:])
