'''
This example wriiten by Marco Lanaro,

"Use Tensorflow DNNClassifier estimator to classify MNIST dataset,"
https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940

code: 
https://gist.github.com/marcolanaro/67b77346730c0862b17c4800ee599286#file-mnist_estimator-py
'''

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],     # two hidden layers (one with 256 neurons, and the other with 32 neurons)
    optimizer=tf.train.AdamOptimizer(1e-4),     # better than gradient descent, ref: https://arxiv.org/pdf/1412.6980.pdf
    n_classes=10,
    dropout=0.1
    #, model_dir="./tmp/mnist_model"
    # save the current model so that the next time can continously train
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

print("Training ...")
classifier.train(input_fn=train_input_fn, steps=1000)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
print("Evaluating ...")
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

'''
step =  100000, Test Accuracy: 98.220003%
step =  10000, Test Accuracy: 96.850002%
step =  1000, Test Accuracy: 91.450000%
'''