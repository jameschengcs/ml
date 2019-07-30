# Data splitting
import numpy as np
import tensorflow as tf
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] # (7, 2)
Y = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]] # (7, 3)

ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size = 7, reshuffle_each_iteration = False)
dsTrain = ds.take(5)     # shuffling ds again, then taking 5 elements from ds
dsTest = ds.skip(5)      # shuffling ds again, then taking 2 elements from ds
dsTest2 = ds.shard(3, 0) # shuffling ds again, then taking 2 elements from ds
itr = ds.make_one_shot_iterator()
elem = itr.get_next()
itrTrain = dsTrain.make_one_shot_iterator()
elemTrain = itrTrain.get_next()
itrTest = dsTest.make_one_shot_iterator()
elemTest = itrTest.get_next()
itrTest2 = dsTest2.make_one_shot_iterator()
elemTest2 = itrTest2.get_next()

with tf.Session() as sess:
    print('Origin')
    for i in range(7):
        print(sess.run(elem))        
    print('Train')
    for i in range(5):
        print(sess.run(elemTrain))
    print('Test')  
    for i in range(2):
        print(sess.run(elemTest))        
    print('Test2')  
    for i in range(2):
        print(sess.run(elemTest2)) 
