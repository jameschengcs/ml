import tensorflow as tf
import numpy as np
W = tf.Variable([[1.0], [1.0], [1.0], [1.0]])
B = tf.Variable([0.0000])

@tf.function
def model(X):
    return tf.matmul(X, W) + B

def optimizeW(Y, M, X, i, rate):    
    global W
    Xi = np.reshape(X[:, i], Y.shape)
    D = Y - M  
    s = tf.reduce_sum(Xi * D)
    U = np.zeros(W.shape)
    U[i] = s * rate    
    W.assign(W + U)
    return D

def optimizeB(Y, M, X, rate):    
    global W
    D = Y - M
    s = tf.reduce_sum(D) 
    B.assign(B + s * rate) 
    return D
    
    
X = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.22, 0.25, 0.32, 0.3],
              [0.35, 0.23, 0.33, 0.2],
              [0.43, 0.25, 0.35, 0.1], 
              [0.53, 0.25, 0.36, 0.0]], dtype = np.float32)
Y = np.array([[0.8], [0.6], [0.4], [0.3], [0.2]], dtype = np.float32)

epochs = 15
numDim = W.shape[0]
r = 1.0
for iE in range(epochs):  
    for iD in range(numDim):
        M = model(X)
        optimizeW(Y, M, X, iD, r) 
    M = model(X)
    D = optimizeB(Y, M, X, r)          
    print('SSE: ', tf.reduce_sum(D * D))      
    r -= 0.01                 
print(model(X))
