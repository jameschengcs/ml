import numpy as np
import tensorflow as tf
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] # (7, 2)
Y = [12, 34, 56, 78, 910, 1112, 1314]

print('from_tensor_slices')
ds = tf.data.Dataset.from_tensor_slices((X, Y))
print(ds)
itr = ds.make_one_shot_iterator()
item = itr.get_next()
with tf.Session() as sess:
    for i in range(7):
        print(sess.run(item))
        
print('from_tensors')
ds = tf.data.Dataset.from_tensors((X, Y))
print(ds)
itr = ds.make_one_shot_iterator()
item = itr.get_next()
with tf.Session() as sess:
    print(sess.run(item))
