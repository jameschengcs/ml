# Repeat & Batch 
import numpy as np
import tensorflow as tf
X = [[1., 2.],     [3., 4.],     [5., 6.],     [7., 8.],     [9., 10.],    [11., 12.],   [13., 14.]] # (7, 2)
Y = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]] # (7, 3)
m = 7
batchSize = 3
batches = m / batchSize
epochs = 4
ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size = batchSize, reshuffle_each_iteration = False).batch(batchSize, drop_remainder = True).repeat(epochs)
itr = ds.make_one_shot_iterator()
elem = itr.get_next()

W = tf.Variable(tf.random.uniform([batchSize, 2], dtype = tf.float32))
T = tf.Variable(tf.random.uniform([batchSize, 3], dtype = tf.float32))
setW = W.assign(elem[0])
setT = T.assign(elem[1])
with tf.Session() as sess:
    for k in range(epochs):
        print('epoch', k)
        for i in range(2):
            sess.run((setW, setT))        
            print(sess.run(W), sess.run(T))   
