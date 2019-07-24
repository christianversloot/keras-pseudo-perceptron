# Load dependencies
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import hard_sigmoid
import numpy as np

# Load data
dataset = np.loadtxt('./pima_dataset.csv', delimiter=',')

# Separate train and test data
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create the Perceptron
model = Sequential()
model.add(Dense(1, input_shape=(8,), activation=hard_sigmoid, kernel_initializer='glorot_uniform'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Perceptron
model.fit(X, Y, epochs=225, batch_size=25, verbose=1, validation_split=0.2)