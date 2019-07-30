import numpy as np
import tensorflow as tf
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] # (7, 2)
Y = [12, 34, 56, 78, 910, 1112, 1314]

print('Original data')
ds = tf.data.Dataset.from_tensor_slices((X, Y))
print(ds)
itr = ds.make_one_shot_iterator()
elem = itr.get_next()
with tf.Session() as sess:
    for i in range(7):
        ins = sess.run(elem)
        print(ins[0], ins[1])
        
print('Shuffled data')        
dss = ds.shuffle(buffer_size = 7) # Don't forget to update the dataset    
itr = dss.make_one_shot_iterator()
elem = itr.get_next()
with tf.Session() as sess:
    for i in range(7):
        print(sess.run(elem))

print('3X Shuffled data')        
ds3 = dss.repeat(3) # Don't forget to update the dataset    
itr = ds3.make_one_shot_iterator()
elem = itr.get_next()
with tf.Session() as sess:
    for i in range(21):
        print(i, ":", sess.run(elem))
     
print('Batched data')    
dsb = ds3.batch(5) 
itr = dsb.make_one_shot_iterator()
elem = itr.get_next()
with tf.Session() as sess:
    for i in range(5):
        B = sess.run(elem)
        print(B[0], B[1])
