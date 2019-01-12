
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils


X= tf.placeholder(tf.float32, shape=[4,2], name = 'input_x')
Y = tf.placeholder(tf.float32, shape=[4,1], name = 'output_y') 

w1 = tf.get_variable(initializer = tf.random_uniform([2,2], -1, 1), name = 'w1')
w2 = tf.get_variable(initializer = tf.random_uniform([2,1], -1, 1), name = 'w2')

b1 = tf.get_variable(initializer = tf.zeros([4,2]), name = '1')
b2 = tf.get_variable(initializer = tf.zeros([4,1]), name = '2') 

# When you are calculating the loss function using tf.nn.sigmoid_cross_entropy_with_logits
# then here's how the NN should look like: 
# layer1 = tf.matmul(X , w1) + b1 
# act = tf.nn.relu(layer1)
# logit = tf.matmul(act, w2) + b2
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = logit)
# And in the last epoch when you want to get the final value of the predicted Y,
# you need to get tf.nn.softmax(logit) 


# When you calculate the loss function yourself, then since the last layer has the sigmoinf
# activation function, you do it like tgis: 
# layer1 = tf.matmul(X , w1) + b1 
# # act = tf.nn.relu(layer1)
# # logit_2 = tf.matmul(act, w2) + b2
# # logit  = tf.nn.sigmoid(logit_2)
# # loss = tf.reduce_mean(( (Y * tf.log(logit)) + ((1 - Y) * tf.log(1.0 - logit)) ) * -1)
# and the last ecoch, value for predicted Y, is simply output of logit 


layer1 = tf.matmul(X , w1) + b1 
act = tf.nn.relu(layer1)
logit = tf.matmul(act, w2) + b2
# logit  = tf.nn.sigmoid(logit_2)


# loss = tf.reduce_mean(( (Y * tf.log(logit)) + ((1 - Y) * tf.log(1.0 - logit)) ) * -1)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = logit)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)

# x = tf.constant([[0,0], [0,1], [1,0], [1,1]], dtype=tf.int32, shape = [4,2], name = 'x')
# y = tf.constant([[0], [1], [1], [0]], dtype=tf.int32, shape = [4,1], name = 'y')

x = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]


n_epochs = 10000
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    # sess.run(print(w1))
    # exit
    for i in range(n_epochs): # train the model n_epochs times
            total_loss = 0
            # print("I am here")
            opt, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y}) 
            total_loss += l
            # print('Average loss epoch {0}: {1}'.format(i, l))
            if i == 9999: 

                print('Hypothesis ', sess.run(tf.sigmoid(logit), feed_dict={X:x, Y: y}))
                # print("w1 = {0}".format(sess.run(w1)))
                # print("w2 = {0}".format(sess.run(w2)))
                # print("b1 = {0}".format(sess.run(b1)))
                # print("b2 = {0}".format(sess.run(b2)))
            # if i == 99999: 

            #     print('Hypothesis ', sess.run(tf.nn.sigmoid(logit), feed_dict={X:x, Y: y}))
            #     print('Hypothesis ', sess.run(loss, feed_dict={X:x, Y: y}))

            #     print("w1 = {0}".format(sess.run(w1)))
            #     print("w2 = {0}".format(sess.run(w2)))
            #     print("b1 = {0}".format(sess.run(b1)))
            #     print("b2 = {0}".format(sess.run(b2)))
                

        
    print('Optimization Finished!')
    # print('total loss== {0}'.format(total_loss/n_epochs)) 


# Result with mannually implemented cross entropy
# Hypothesis  [[0.03879504]
#  [0.98889285]
#  [0.98889846]
#  [0.03879398]]
# Hypothesis  0.025366765
# w1 = [[-2.3746912  2.3391335]
#  [ 2.3746812 -2.3391898]]
# w2 = [[3.242076]
#  [3.29164 ]]
# b1 = [ 8.7064163e-06 -4.6647245e-05]
# b2 = [-3.2099235]
# Optimization Finished!
# total loss== 2.536667324602604e-07


# Result using tf's loss function
# Hypothesis  [[0.00100505]
#  [0.998995  ]
#  [0.9999515 ]
#  [0.00100505]]
# Hypothesis  [[1.0055506e-03]
#  [1.0055506e-03]
#  [4.8458332e-05]
#  [1.0055506e-03]]
# w1 = [[ 1.7365735 -0.3560791]
#  [-1.3467168 -0.7698252]]
# w2 = [[ 2.6038923]
#  [-0.7432482]]
# b1 = [[ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 1.6136835   0.        ]
#  [-0.38988867  0.        ]]
# b2 = [[-6.901717 ]
#  [ 6.901717 ]
#  [ 1.2110736]
#  [-6.901717 ]]
# Optimization Finished!
# total loss== [[1.00556074e-08]
#  [1.00556074e-08]
#  [4.84589800e-10]
#  [1.00556074e-08]]