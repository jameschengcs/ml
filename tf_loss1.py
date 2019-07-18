import tensorflow as tf
import numpy as np
Xs = [1 ,2, 3, 4, 5, 6]
Ys = [0, 4, 1, 8, 2, 1]

D = np.array(Xs) - np.array(Ys)
Dmse = np.sum(np.square(D)) / D.size
Dabs = np.sum(np.abs(D)) / D.size
print(Dmse, Dabs)

# Data initialization
X = tf.placeholder(dtype = tf.float32, shape = [6])
Y = tf.placeholder(dtype = tf.float32, shape = [6])

# Operation definition
lossMSE = tf.losses.mean_squared_error(X, Y)
lossABS = tf.losses.absolute_difference(X, Y)

# Initializing a session & Running 
with tf.Session() as sess:
    print(sess.run(lossMSE, feed_dict = {X: Xs, Y: Ys}))
    print(sess.run(lossABS, feed_dict = {X: Xs, Y: Ys}))
    
