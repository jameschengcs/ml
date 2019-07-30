import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

Iris = load_iris()
m, n = np.shape(Iris.data) # n records of m features
k = Iris.target_names.shape[0]
test_ratio = 0.2
train_ratio = 1.0 - test_ratio

print(m, n, k)
print(Iris.feature_names)
print(Iris.target_names)
#print(Iris.data[0])

Ys = np.zeros(shape = [m, k])
for i in range(m):
    Ys[i][Iris.target[i]] = 1

nItem = 10
nTrain = int(m * train_ratio)
nTest = int(m * test_ratio)
nBatch = nTrain // nItem;
print(nTrain, nTest, nBatch)

# data initialization
X = tf.placeholder(dtype = tf.float32, shape = [nItem, n])
Y = tf.placeholder(dtype = tf.float32, shape = [nItem, k])
XT = tf.placeholder(dtype = tf.float32, shape = [nTest, n])
YT = tf.placeholder(dtype = tf.float32, shape = [nTest, k])

# Set model weights
W = tf.Variable(tf.random.uniform([n, k], dtype = tf.float32))
B = tf.Variable(tf.random.uniform([k], dtype = tf.float32))
E = tf.Variable([0.0])
                                                
# Operation definition
WBTrain = tf.nn.xw_plus_b(X, W, B)
loss = tf.losses.sigmoid_cross_entropy(Y, WBTrain)
learning_rate = 0.1
#opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)                                            
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)            
init = tf.global_variables_initializer()

WBTest = tf.nn.xw_plus_b(XT, W, B)
lossTest = tf.losses.sigmoid_cross_entropy(YT, WBTest)
predTest = tf.sigmoid(WBTest)

nEpoch = 10

with tf.Session() as sess:
    sess.run(init)   
    for epoch in range(nEpoch):
        X_train, X_test, Y_train, Y_test = train_test_split(Iris.data, Ys, test_size = test_ratio)        
        Y_test_t = np.argmax(Y_test, axis = 1)
        for i in range(nBatch):
            iT = i * nItem
            Xb = X_train[iT: iT + nItem, :]
            Yb = Y_train[iT: iT + nItem, :]

            sess.run(opt, feed_dict={X: np.reshape(Xb, [nItem, n]), Y: np.reshape(Yb, [nItem, k])})
            error = sess.run(lossTest, feed_dict={XT: np.reshape(X_test, [nTest, n]), YT: np.reshape(Y_test, [nTest, k])})
            P = sess.run(predTest, feed_dict={XT: np.reshape(X_test, [nTest, n])})
            print(epoch, '-', i, 'error: ', error)
            Pt = np.argmax(P, axis = 1)
            print(Pt)
            print(Y_test_t)
            print(np.count_nonzero(Pt==Y_test_t))     
