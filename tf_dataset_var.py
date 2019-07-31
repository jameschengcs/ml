# Data > variable
import numpy as np
import tensorflow as tf
X = [[1., 2.],     [3., 4.],     [5., 6.],     [7., 8.],     [9., 10.],    [11., 12.],   [13., 14.]] # (7, 2)
Y = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]] # (7, 3)
T = np.array([[[1, 2], [1, 0, 1]]])
print(T.shape)
ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size = 7, reshuffle_each_iteration = False)
itr = ds.make_one_shot_iterator()
elem = itr.get_next()

W = tf.Variable(tf.random.uniform([2], dtype = tf.float32))
T = tf.Variable(tf.random.uniform([3], dtype = tf.float32))
setW = W.assign(elem[0])
setT = T.assign(elem[1])
with tf.Session() as sess:
    for i in range(7):`
        sess.run((setW, setT))    # Assignments of elem[0] and elem[1] must be in a sess.run
        print(sess.run(W), sess.run(T))     
