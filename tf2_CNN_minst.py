import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Conv2D   
from tensorflow.keras.layers import MaxPooling2D  
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense 

mnist = tf.keras.datasets.mnist
(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
Xtrain = np.reshape(Xtrain / 255.0, (60000,28, 28, 1))
Xtest = np.reshape(Xtest / 255.0, (10000,28, 28, 1))
Ytrain = tf.keras.utils.to_categorical(Ytrain)
Ytest = tf.keras.utils.to_categorical(Ytest)

init = tf.keras.initializers.Zeros()
model = tf.keras.models.Sequential()

model.add(Conv2D(16, 8, strides=2, kernel_initializer = init, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, 4, strides=2, kernel_initializer = init, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_initializer = init))
model.add(Dense(10, activation='softmax', kernel_initializer = init))


model.compile(optimizer='sgd', loss = "categorical_crossentropy", metrics=['categorical_accuracy'] )

model.fit(Xtrain, Ytrain, batch_size = 100, epochs = 5)
model.evaluate(Xtest,  Ytest, verbose = 2)
