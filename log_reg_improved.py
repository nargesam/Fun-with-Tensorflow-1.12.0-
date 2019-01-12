######### Changes made to the model ###############

## The architecture of the model has been changed compared to the pervious question

## A new hidden layer with size 100 is introduced
## Updated Architecture:
# 1. Input Layer                       - batch size x total_pixel_size
# 2. Hidden Layer with relu activation - total_pixel_size x 100
# 3. Output Layer                      - 100 x 10 (total label size) 

## Epoch size has been increased to 100

## Learning rate maintained at 0.01

###### Final accuracy 97.52%

#####################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 100
n_train = 60000
n_test = 10000


mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)


# Step 1: Read in data
# mnist_folder = os.getcwd()+'\\data\\'
# utils.download_mnist(mnist_folder)  - Manually downloaded thef files and extracted them
# train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

#############################
########## TO DO ############
#############################
# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000) # if you want to shuffle your data
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
#############################
########## TO DO ############
#############################

hidden_layer_size = 100

### Creating a hidden layer with a size of 50
w1 = tf.get_variable(initializer=tf.random_normal(shape = [img.shape[1].value,hidden_layer_size],
                                                 mean = 0, stddev= 0.01),name = 'Weight_1')

b1 = tf.get_variable(initializer=tf.zeros(shape = [hidden_layer_size]), name = 'Bias_1')

layer_1 = tf.add(tf.matmul(img,w1), b1)
layer_1_act = tf.nn.relu(layer_1)


######## Output Layer
w2 = tf.get_variable(initializer=tf.random_normal(shape = [hidden_layer_size,label.shape[1].value],
                                                 mean = 0, stddev= 0.01),name = 'Weight_2')



b2 = tf.get_variable(initializer=tf.zeros(shape = [label.shape[1].value]), name = 'Bias_2')

logits = tf.add(tf.matmul(layer_1_act,w2), b2)



# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
#############################
########## TO DO ############
#############################
loss =  tf.nn.softmax_cross_entropy_with_logits_v2(labels = label,logits = logits)

# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
#############################
########## TO DO ############
#############################
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)



# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except ValueError:
            pass
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()