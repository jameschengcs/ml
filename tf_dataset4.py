import numpy as np
import tensorflow as tf
X1 = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] # (7, 2)
Y1 = [12, 34, 56, 78, 910, 1112, 1314]
X2 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.10], [0.11, 0.12], [0.13, 0.14]] # (7, 2)
Y2 = [0.12, 0.34, 0.56, 0.78, 0.91, 0.1112, 0.1314]

X = tf.placeholder(dtype=tf.float32, shape=[7, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[7])
ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size = 10)

itr = ds.make_initializable_iterator()
elem = itr.get_next()
with tf.Session() as sess:
    print('Dataset 1')    
    sess.run(itr.initializer, feed_dict={X: X1, Y: Y1})
    for i in range(7):
        print(sess.run(elem))
    print('Dataset 2')    
    sess.run(itr.initializer, feed_dict={X: X2, Y: Y2})
    for i in range(7):
        print(sess.run(elem))
