
# Load pickled data

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.image as img
import glob as glob
import cv2
print ("Dependencies successfully loaded !")
################################################################################

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print ("Data unpickled !")
################################################################################
## Image preprocessing pipeline
def imagePreprocess(image_set):

    img_train = image_set
    len_train = len(img_train)
    # convert to grayscale : (r+g+b / 3)
    X_gray = np.sum(img_train/3, axis=3, keepdims=True)
    # normalize
    X_norm = (X_gray - 128)/128
    return X_norm

print ("Image preprocessing pipeline defined !")
################################################################################

X_preprocess_train = imagePreprocess(X_train)
X_preprocess_test = imagePreprocess(X_test)
X_preprocess_valid = imagePreprocess(X_valid)

X_test = X_preprocess_test
X_train = X_preprocess_train
X_valid = X_preprocess_valid

# Shuffle Training Set
X_train, y_train = shuffle(X_train, y_train)

print ("Train, test and valid dataset preprocessing complete !")
################################################################################

# Arguments used for tf.truncated_normal,
# randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# argument shape = (patch_height, patch_width, input_depth, output_depth)

weights = {
    'wc1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)),
    'wc2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
    'fc1' : tf.Variable(tf.truncated_normal(shape=(400,120), mean = mu, stddev = sigma)),
    'fc2' : tf.Variable(tf.truncated_normal(shape=(120,84), mean = mu, stddev = sigma)),
    'fc3' : tf.Variable(tf.truncated_normal(shape=(84,43), mean = mu, stddev = sigma))
}

bias = {
    'wc1' : tf.Variable(tf.zeros(6)),
    'wc2' : tf.Variable(tf.zeros(16)),
    'fc1' : tf.Variable(tf.zeros(120)),
    'fc2' : tf.Variable(tf.zeros(84)),
    'fc3' : tf.Variable(tf.zeros(43))
}
################################################################################

def LeNet(x):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer_1 = tf.nn.conv2d(x, weights['wc1'], strides=[1,1,1,1], padding='VALID')
    layer_1 = tf.nn.bias_add(layer_1, bias['wc1'])
    print("Layer 1 shape :", layer_1.get_shape())

    # Activation.
    layer_1 = tf.nn.relu(layer_1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1] , strides=[1,2,2,1], padding='VALID')
    print("Layer 1 pooling shape :", layer_1.get_shape())

    # Layer 2: Convolutional. Output = 10x10x16.
    layer_2 = tf.nn.conv2d(layer_1, weights['wc2'], strides=[1,1,1,1], padding='VALID')
    layer_2 = tf.nn.bias_add(layer_2, bias['wc2'])
    print("Layer 2 shape :", layer_2.get_shape())

    # Activation.
    layer_2 = tf.nn.relu(layer_2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print("Layer 2 pooling shape :", layer_2.get_shape())

    # Flatten. Input = 5x5x16. Output = 400.
    layer_f = flatten(layer_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    layer_3 = tf.add(tf.matmul(layer_f, weights['fc1']), bias['fc1'])

    # Activation.
    layer_3 = tf.nn.relu(layer_3)

    # Dropout
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    layer_4 = tf.add(tf.matmul(layer_3, weights['fc2']), bias['fc2'])

    # Activation.
    layer_4 = tf.nn.relu(layer_4)

    # Dropout
    layer_4 = tf.nn.dropout(layer_4, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(layer_4, weights['fc3']), bias['fc3'])
    print("Logits shape :", logits)

    #return logits
    return logits

print ("LeNet architecture defined !")
################################################################################

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
################################################################################

rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
################################################################################

EPOCHS = 10
BATCH_SIZE = 100

print ("epochs :", EPOCHS)
print ("batch size :", BATCH_SIZE)
################################################################################

## tf.argmax Returns the index with the largest value across axes of a tensor
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
## Computes the mean of elements across dimensions of a tensor.
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
################################################################################

print("Loading set of images collected from web...")

imageWeb = np.zeros((len(glob.glob('./my_data/*.jpg')),32,32,3))
loc = [x for x in glob.glob("./my_data/*.jpg")]
for i in range(len(loc)):
    temp = cv2.imread(loc[i])
    imageWeb[i] = temp[:,:,0:3]

print("Number of Images :",len(imageWeb))

k = 331 #for set of 9 or less images
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in imageWeb:
    plt.subplot(k)
    plt.imshow(i, cmap="gray")
    k += 1
plt.show()

labelWeb = [14,28,8,18,13]

k = 331

plt.subplots_adjust(wspace=0.5, hspace=0.5)
for yy in labelWeb:
    k_value = np.min(np.argwhere(y_train==yy))
    plt.subplot(k)
    plt.imshow(X_train[k_value].squeeze(), cmap='gray')
    k += 1
    plt.title(y_train[k_value])
plt.show()

imageWeb = imagePreprocess(imageWeb)

print("Image Set Dimension :", imageWeb.shape)
print("Label Dimension :", len(labelWeb))

################################################################################

print ("Running validation on web data...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_valid = tf.train.import_meta_graph('./lenet.meta')
    saver_valid.restore(sess, "./lenet")
    accuracy_of_set = evaluate(imageWeb, labelWeb)
    print("Accuracy of web test set = {:.3f}".format(accuracy_of_set))

################################################################################
# Analyze Performance
print("Total number of Images :", len(imageWeb))
print("Predicted correctly : ", int(accuracy_of_set*len(imageWeb)))
print("Accuracy of web test set = {:.3f}".format(accuracy_of_set))
print("Softmax probability display : ")
k = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_valid = tf.train.import_meta_graph('./lenet.meta')
    saver_valid.restore(sess, "./lenet")
    valid_set_logits = sess.run(softmax_logits,
                               feed_dict={
                                   x: imageWeb,
                                   keep_prob: 1.0
                               })
    valid_set_topk = sess.run(top_k,
                             feed_dict={
                                 x: imageWeb,
                                 keep_prob: 1.0
                             })

    location = 1
    row = k+1
    for i in range(len(imageWeb)):
        plt.subplot(1,row,location)
        plt.title("input" + str(i+1))
        plt.imshow(imageWeb[i].squeeze(), cmap="gray")

        for j in range(5):
            guess = valid_set_topk[1][i][j]
            index = np.argwhere(y_train == guess)[0]
            location += 1
            plt.subplot(1,row,location)
            percentage = valid_set_topk[0][i][j]*100
            title = ('Guess {} \n {:.0f}%').format(j+1, percentage)
            plt.title(title)
            plt.imshow(X_train[index].squeeze(), cmap="gray")
        location = 1
        plt.show()
