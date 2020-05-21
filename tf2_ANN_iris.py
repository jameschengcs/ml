import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import tensorflow as tf

iris = load_iris()

initializer = tf.keras.initializers.Zeros()
model = tf.keras.models.Sequential([  
    tf.keras.layers.Dense(10, activation='linear', kernel_initializer = initializer), 
    tf.keras.layers.Dense(5, activation='relu', kernel_initializer = initializer),
    tf.keras.layers.Dense(3, activation='sigmoid', kernel_initializer = initializer)    
])

model.compile(optimizer='sgd',
              loss= tf.keras.losses.mean_squared_error,
              metrics=['mse'])

X = iris.data 
Y = iris.target 
X, Y = shuffle(X, Y)
Xtrain = X[:130, :]
Ytrain = Y[:130]
Ytrain = tf.keras.utils.to_categorical(Ytrain)
print(Ytrain.shape)
model.fit(Xtrain, Ytrain, batch_size = 1, epochs = 10, verbose = 1) 

Xtest = X[130:, :]
Ytest = Y[130:]
Ytest = tf.keras.utils.to_categorical(Ytest)
model.evaluate(Xtest,  Ytest, verbose=0)
