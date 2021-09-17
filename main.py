# Import the required libraries
from __future__ import print_function
import os
import shutil
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import tkinter
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

# Loading the Images from the image folder
path = "./datasets/images/"

filenames = [f for f in listdir(path) if isfile(join(path, f))]

print(str(len(filenames)) + " images loaded")

# Splitting the data into training and test datasets
dogs_count = 0
cats_count = 0
training_size = 1000
test_size = 500
training_images = []
training_labels = []
test_images = []
test_labels = []
size = 150
dogs_train = "./datasets/catsvsdogs/train/dogs/"
cats_train = "./datasets/catsvsdogs/train/cats/"
dogs_validation = "./datasets/catsvsdogs/validation/dogs/"
cats_validation = "./datasets/catsvsdogs/validation/cats/"


# A function to create the directory if it doesn't exist
def create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


create_directory(dogs_train)
create_directory(cats_train)
create_directory(dogs_validation)
create_directory(cats_validation)


# A function to get the zeros from the filename of Image
def get_zeros(number):
    if 10 < number < 100:
        return '0'
    if number < 10:
        return '00'
    else:
        return ''


# loop through all the images to split the data
for i, file in enumerate(filenames):

    if filenames[i][0] == "d":
        dogs_count += 1
        image = cv2.imread(path + file)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        if dogs_count <= training_size:
            training_images.append(image)
            training_labels.append(1)
            zeros = get_zeros(dogs_count)
            cv2.imwrite(dogs_train + "dog" + str(zeros) + str(dogs_count) + ".jpg", image)
        if training_size < dogs_count <= training_size + test_size:
            test_images.append(image)
            test_labels.append(1)
            zeros = get_zeros(dogs_count - 1000)
            cv2.imwrite(dogs_validation + "dog" + str(zeros) + str(dogs_count - 1000) + ".jpg", image)

    if filenames[i][0] == "c":
        cats_count += 1
        image = cv2.imread(path + file)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        if cats_count <= training_size:
            training_images.append(image)
            training_labels.append(0)
            zeros = get_zeros(cats_count)
            cv2.imwrite(cats_train + "cat" + str(zeros) + str(cats_count) + ".jpg", image)
        if training_size < cats_count <= training_size + test_size:
            test_images.append(image)
            test_labels.append(0)
            zeros = get_zeros(cats_count - 1000)
            cv2.imwrite(cats_validation + "cat" + str(zeros) + str(cats_count - 1000) + ".jpg", image)

    if dogs_count == training_size + test_size and cats_count == training_size + test_size:
        break

print("Training and Test Datasets are ready.")

# Using the e savez function from numpy to save the datasets as arrays in the npz files
np.savez("cats_vs_dogs_training_data.npz", np.array(training_images))
np.savez("cats_vs_dogs_training_labels.npz", np.array(training_labels))
np.savez("cats_vs_dogs_test_data.npz", np.array(test_images))
np.savez("cats_vs_dogs_test_labels.npz", np.array(test_labels))


# Loading the datasets that we have created
def load_data(datasetname):
    npzfile = np.load(datasetname + "_training_data.npz")
    train = npzfile['arr_0']

    npzfile = np.load(datasetname + "_training_labels.npz")
    train_labels = npzfile['arr_0']

    npzfile = np.load(datasetname + "_test_data.npz")
    test = npzfile['arr_0']

    npzfile = np.load(datasetname + "_test_labels.npz")
    test_labels = npzfile['arr_0']

    return (train, train_labels), (test, test_labels)


# To display few images from the training dataset
# for i in range(1, 11):
#     random = np.random.randint(0, len(training_images))
#     cv2.imshow("image_"+str(i), training_images[random])
#     if training_labels[random] == 0:
#         print(str(i) + " - Cat")
#     else:
#         print(str(i) + " - Dog")
#     cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# Reshaping the data
# Load data
(x_train, y_train), (x_test, y_test) = load_data("cats_vs_dogs")

# Reshaping the label data from (2000,) to (2000,1) and test data from (1000,) to (1000,1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Change the images to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the data
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

batch_size = 16
epochs = 25

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols, 3)

# Creating the Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

# Training the Model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)

model.save('./cats-dogs-cnn-v1.h5')

# Evaluating the Performance of the trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test Loss:', scores[0])
print('Test Accuracy:', scores[1])

# Classifying the Images and loading the model
classifier = load_model('./cats-dogs-cnn-v1.h5')


# To display the prediction along with the image of cat/dog
def draw(name, pred, input_image):
    BLACK = [0, 0, 0]
    if pred == '[0]':
        pred = 'cat'
    if pred == '[1]':
        pred = 'dog'
    expanded_image = cv2.copyMakeBorder(input_image, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, str(pred), (252, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow(name, expanded_image)


# looping through the test dataset to get 10 random images
for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    input_image = x_test[rand]
    imageL = cv2.resize(input_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('Test Image', imageL)

    input_image = input_image.reshape(1, 150, 150, 3)

    # Prediction
    res = str(classifier.predict_classes(input_image, 1, verbose=0)[0])
    # print(res)
    # if res == '[0]':
    #     print("Cat")
    # if res == '[1]':
    #     print("Dog")
    draw('Prediction', res, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()
