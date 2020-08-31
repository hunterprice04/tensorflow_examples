#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow import keras
print("djfhvk")
"""
Keras is an API in TensorFlow that makes it easy to define neural networks.
In keras you use the word Dense to define a layer of connected neurons.
Here, there is only one dense so there is one layer, and units is equal
to 1 so there is only one neuron.
Successive layers are defined in sequence hence the word Sequential.
"""
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

"""
There are two major function rules to be aware of: Loss functions and Optimizers.
The Loss function calculates how good or bad a guess is based off the previous 
data its been given.
The Loss function measures the loss then gives the data to the optimizer which gives 
the next guess
Here, the loss function is mean squared error and the optimizer is stochastic 
gradient descent (sgd).
"""
model.compile(optimizer="sgd", loss='mean_squared_error')

"""
this is the training data that we are passing to the model 
Y = 2X + 1
"""
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

"""
Training takes place in the fit command. It fits the model to the data.
epochs is the number of times the model will go through the training loop.
The training loop consists of making a guess, measuring how good or bad 
the guesses are (loss function), then use the optimizer and the data to 
make another guess.
Here, we are asking the model to fit the X values to the Y values.
"""
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
