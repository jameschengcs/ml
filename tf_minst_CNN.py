import numpy as np
import tensorflow as tf

T = tf.float32
Tnp = np.float32
classes = 10
def readInt32(f, byteorder = 'big'):
    return int.from_bytes(f.read(4), byteorder)

def loadLabel(filepath):
    with open(filepath, "rb") as f:
        magic = readInt32(f)
        if magic == 2049:
            n = readInt32(f)
            a = np.fromfile(f, dtype = np.ubyte)
            b = np.zeros(shape = [n, classes], dtype = Tnp )
            for i in range(n):
                b[i, a[i]] = 1.0            
            return b;
        
def loadImage(filepath):
    with open(filepath, "rb") as f:
        magic = readInt32(f)       
        if magic == 2051:
            n = readInt32(f)
            h = readInt32(f)
            w = readInt32(f)            
            a = np.fromfile(f, dtype = np.ubyte).astype(np.float32) / 255.0
            return np.reshape(a,[n, h, w, 1]);
        
def varInitC(shape, value = 0.0):
    return tf.constant(value = value, shape = shape, dtype = T)
def varInitN(shape, stddev = 0.05):
    return tf.truncated_normal(shape = shape, dtype = T, stddev=stddev)
def varInitU(shape, minval = -0.1, maxval = 0.1):
    return tf.random.uniform(shape = shape, minval = minval, maxval = maxval, dtype = T)

def createVarW(shape):
    return tf.Variable(varInitN(shape))
    #return tf.Variable(varInitU(shape))
def createVarB(shape):
    #return tf.Variable(varInitU(shape))
    return tf.Variable(varInitC(shape))

    
imgTrain = loadImage("mnist/train-images-idx3-ubyte")
lbTrain = loadLabel("mnist/train-labels-idx1-ubyte")        
imgTest = loadImage("mnist/t10k-images-idx3-ubyte")
lbTest = loadLabel("mnist/t10k-labels-idx1-ubyte")

nTrain, H, W, CH = imgTrain.shape
nTest = imgTest.shape[0]
batcheSize = 20
batches = nTrain // batcheSize
batchesT = nTest // batcheSize
loadBatches = 5
loadRounds = batches // loadBatches
loadRoundsT = batchesT // loadBatches
loadImages = loadBatches * batcheSize

XB = tf.placeholder(shape = [loadImages, W, H, CH], dtype = T)
YB = tf.placeholder(shape = [loadImages, classes], dtype = T)
X = tf.Variable(tf.zeros([batcheSize, W, H, CH], dtype = T))
Y = tf.Variable(tf.zeros([batcheSize, classes], dtype = T))

ds = tf.data.Dataset.from_tensor_slices((XB, YB))
ds = ds.shuffle(buffer_size = loadImages)
ds = ds.batch(batch_size = batcheSize)
itr = ds.make_initializable_iterator()
elem = itr.get_next()
setXY = (X.assign(elem[0]), Y.assign(elem[1])) #[[W, H, CH]] => [1, W, H, CH]; [[10]] => [1, 10]
                                    
# Convolution layer 1 

W1 = createVarW([5, 5, 1, 16]) # 8 5x5 kernels
B1 = createVarB([16]) # 32 Biases
cnn1_conv = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')    
cnn1_act = tf.nn.relu(cnn1_conv + B1)
cnn1 = tf.nn.max_pool(cnn1_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# the shape of cnn1 is (batcheSize, W/2, H/2, 32)

# Convolution layer 2
W2 = createVarW([3, 3, 16, 32]) # 3x3 kernels 8 -> 16 
B2 = createVarB([32]) # 64 Biases
cnn2_conv = tf.nn.conv2d(cnn1, W2, strides = [1, 1, 1, 1], padding = 'SAME')    
cnn2_act = tf.nn.relu(cnn2_conv + B2)
cnn2 = tf.nn.max_pool(cnn2_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# the shape of cnn2 is (batcheSize, W/4, H/4, 64)

# Fully connection layer 1
nf = 7 * 7 * 32
W3 = createVarW([nf, 256])
B3 = createVarB([256])  
fcX = tf.reshape(cnn2, [-1, nf])
fc1_xwb = tf.nn.xw_plus_b(fcX, W3, B3)
fc1_act = tf.nn.relu(fc1_xwb)
fc1 = tf.nn.dropout(fc1_act, keep_prob = 0.7)

# Fully connection layer 2
W4 = createVarW([256, 10])
B4 = createVarB([10])  
prediction = tf.nn.softmax(tf.nn.xw_plus_b(fc1, W4, B4))

# Loss function
loss = tf.losses.softmax_cross_entropy(Y, prediction)
#loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction)))       # loss
learning_rate = 0.001 
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
#opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 

# Running
epochs = 15
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    
    iBs = 0
    iBe = loadImages    
    error = 0.0
    nB = loadRounds
    nBT = loadRoundsT
    for iB in range(nBT):
        sess.run(itr.initializer, feed_dict={XB: imgTest[iBs:iBe, :, :, :], YB: lbTest[iBs:iBe ]})            
        errorB = 0.0
        for iE in range(loadBatches):
            sess.run(setXY)
            errorB += sess.run(loss)
        error += (errorB / loadBatches)
        iBs = iBe
        iBe += loadImages
    error /= nBT                    
    print('Init error:', error)  
    
    for epoch in range(epochs):
        print('Training epoch:', epoch)
        iBs = 0
        iBe = loadImages
        for iB in range(nB):
            sess.run(itr.initializer, feed_dict={XB: imgTrain[iBs:iBe, :, :, :], YB: lbTrain[iBs:iBe ]})
            for iE in range(loadBatches):
                sess.run(setXY)
                sess.run(opt)            
            iBs = iBe
            iBe += loadImages
            #if iB % 100 == 0:
            #    print('Batch', iB, 'trained')
        iBs = 0
        iBe = loadImages
        error = 0.0
        for iB in range(nBT):
            sess.run(itr.initializer, feed_dict={XB: imgTest[iBs:iBe, :, :, :], YB: lbTest[iBs:iBe ]})            
            errorB = 0.0
            for iE in range(loadBatches):
                sess.run(setXY)
                errorB += sess.run(loss)
            error += (errorB / loadBatches)
            iBs = iBe
            iBe += loadImages
        error /= nBT                    
        print('Epoch:', epoch, 'error:', error)  
    
