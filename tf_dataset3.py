import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution() # it must be called at program startup.

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]] # (7, 2)
Y = [12, 34, 56, 78, 910, 1112, 1314]

ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size = 1000).repeat(3).batch(3)
for elem in tfe.Iterator(ds):
    print(elem)  
