import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical


## Overview
# This code runs a CNN model (deep learning) on MNIST dataset and renders model summary, test accuracy and test loss as output
# Please check the README file for helpful notes.

# Function to visualise the data.
def data_visualisation(data):
    plt.imshow(data, cmap=plt.cm.binary)
    plt.show()


def modeling(train_images, train_labels, test_images, test_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels,
              batch_size=10,
              epochs=1,
              verbose=1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return test_acc, test_loss


def data_set():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels


def data_preproc(train_images, train_labels, test_images, test_labels):
    train_images = train_images.reshape(
        (60000, 28, 28, 1))  # Converting every image to 1d; train_images has a shape of 60000x28x28
    train_images = train_images.astype('float32') / 255  # scaling between 0,1
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)  # Converts list labels to numpy array (one-hot encoding)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = data_set()
    # data_visualisation(train_images[22])
    train_images, train_labels, test_images, test_labels = data_preproc(train_images, train_labels, test_images,
                                                                        test_labels)
    test_acc, test_loss = modeling(train_images, train_labels, test_images, test_labels)
    print('Test Accuracy- ', test_acc)
    print('Test Loss= ', test_loss)
