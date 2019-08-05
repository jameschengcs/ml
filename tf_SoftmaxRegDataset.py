import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

Iris = load_iris()
m, n = np.shape(Iris.data) # n records of m features
k = Iris.target_names.shape[0]

print(m, n, k)
print(Iris.feature_names)
print(Iris.target_names)
#print(Iris.data[0])

Xs = np.array(Iris.data, dtype = np.float32)
Ys = np.zeros(shape = [m, k], dtype = np.float32)
for i in range(m):
    Ys[i][Iris.target[i]] = 1.0

batchSize = 10
batches = m // batchSize
trainBatches = int(batches * 0.8)
testBatches = batches - trainBatches
epochs = 10
nTrain = trainBatches * batchSize
nTest = testBatches * batchSize
print(nTrain, nTest, trainBatches, testBatches)

# Data initialization
ds = tf.data.Dataset.from_tensor_slices((Xs, Ys))
ds = ds.shuffle(buffer_size = m)
ds = ds.repeat(epochs)
ds = ds.batch(batchSize)
itr = ds.make_one_shot_iterator()
elem = itr.get_next()

# Varaibles
X = tf.Variable(tf.zeros([batchSize, n]), dtype = tf.float32)
Y = tf.Variable(tf.zeros([batchSize, k]), dtype = tf.float32)
W = tf.Variable(tf.random.uniform([n, k], dtype = tf.float32))
B = tf.Variable(tf.random.uniform([k], dtype = tf.float32))
E = tf.Variable([0.0])
                                                
# Data definition
setX = X.assign(elem[0])
setY = Y.assign(elem[1])
xwb = tf.nn.xw_plus_b(X, W, B)
loss = tf.losses.softmax_cross_entropy(Y, xwb)

predTest = tf.nn.softmax(xwb)
learning_rate = 0.1
#opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)                                            
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)  
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)   
    for epoch in range(epochs):
        for i in range(trainBatches):
            sess.run((setX, setY))
            sess.run(opt)
            
        error = 0.0    
        Pt = np.array([])
        Yt = np.array([])
        for i in range(testBatches):    
            sess.run((setX, setY))
            error += sess.run(loss)            
            P = sess.run(predTest)
            Pt = np.append(Pt, np.argmax(P, axis = 1))
            Ytv = sess.run(Y)
            Yt = np.append(Yt, np.argmax(Ytv, axis = 1))
        error /= testBatches    
        print ('Epoch: ', epoch, 'error: ', error)
        print('P:', Pt)
        print('Y:', Yt)
        print(np.count_nonzero(Pt==Yt))     
        
