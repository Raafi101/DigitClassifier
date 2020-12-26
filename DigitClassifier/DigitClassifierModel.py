#Neural Network Model for digit classification

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from cv2 import cv2

(XTrain, yTrain), (XTest, yTest) = keras.datasets.mnist.load_data()

XTrainFlat = XTrain.reshape(len(XTrain), 28 * 28) / 255
XTestFlat = XTest.reshape(len(XTest), 28 * 28) / 255

model = keras.Sequential([
    keras.layers.Dense(28 * 28, input_shape = (28 * 28,) , activation = 'sigmoid'),
    keras.layers.Dense(10, input_shape = (28 * 28,) , activation = 'sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
)

model.fit(XTrainFlat, yTrain, epochs = 15)

yPredicted = model.predict(XTestFlat)

'''
image = cv2.imread('DigitNNEight.png', cv2.IMREAD_GRAYSCALE)

imageFlat = image.reshape(-1, 784) / 255

plt.imshow(image, cmap = 'gray')
plt.show()

print(np.argmax(model.predict(imageFlat)))
'''

'''
model.save('DigitNNModel')
'''