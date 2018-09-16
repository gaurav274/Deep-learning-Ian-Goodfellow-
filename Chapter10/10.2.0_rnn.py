#!/usr/bin/env python
# coding: utf-8

#Adapted from https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

#INPUT
#Random sequence of 0,1

#OUTPUT
#each index has a base probablity of 50% of being 1. If x(i-3) = 1, chances of 1 increases by 50%; but if x(i-8) = 1, it decreases by 25%

##Hyperparameters
hidden_state_size = 5
batch_size = 200 
num_of_cells  = 5   #governs the unrolled length of rnn
num_classes = 2 #vocab size
learning_rate = 0.1


#Generating dataset
import numpy as np
def generate_data(size = 100000):
    X = np.random.choice(2,size)
    Y = []
    for i in range(size):
        threshold = 0.5; #50%chances of being 1
        if((i-3 >= 0) and X[i-3] == 1): #Since our rnn won't be seeing sequence in circular order so checking i-3
            threshold += 0.5;
        elif((i-8) >=0 and X[i-8] == 1):
            threshold -= 0.25;
        
        Y.append(np.random.choice(2,1,p=[1 - threshold, threshold])[0]) #threshold being the chances of 1
    return X, np.array(Y)

def generate_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)
    
    
def generate_epoch(n):
    for i in range(n):
        yield generate_batch(generate_data(), batch_size, num_of_cells)

        
import tensorflow as tf

tf.reset_default_graph()
#Model
#Page 396 RNN


x = tf.placeholder(tf.int32, shape = (batch_size, num_of_cells))
y = tf.placeholder(tf.int32, shape = (batch_size, num_of_cells))
intial_state = tf.zeros((batch_size, hidden_state_size))
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1) # to help iterate over input along num_of_cells


#define weights
with tf.variable_scope("rnn_cell"):
    W = tf.get_variable('W', shape = (hidden_state_size, hidden_state_size)) 
    U = tf.get_variable('U', shape = (num_class, hidden_state_size))
    b = tf.get_variable('b', shape = (hidden_state_size), initializer = tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope("rnn_cell", reuse = True):
        W = tf.get_variable('W', shape = (hidden_state_size, hidden_state_size))
        U = tf.get_variable('U', shape = (num_class, hidden_state_size))
        b = tf.get_variable('b', shape = (hidden_state_size), initializer = tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(rnn_input, U) + tf.matmul(state, W) + b)


#Add rnn_cells to our graph
state = intial_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]    

#Add Loss to our graph
with tf.variable_scope('loss'):
    V = tf.get_variable('V', shape = (hidden_state_size, num_classes))
    c = tf.get_variable('c', shape = (num_classes))

logits = [tf.matmul(rnn_output, V) + c for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]
#unstack ground truth values
y_label = tf.unstack(y, num=num_of_cells, axis = 1)
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = logit) for label,logit in zip(y_label, logits)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

def train_network():
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', s.graph)
        writer.close()
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(generate_epoch(1)):
            for (X,Y) in epoch:
                training_state = np.zeros((batch_size, hidden_state_size))
                loss = 0.0
                training_state, loss_, _ = sess.run([final_state, total_loss, train_step], feed_dict = {x:X, y:Y, intial_state:training_state})
                loss += loss_

                if idx%100 == 0:
                    training_losses.append(loss/100.0)
                    print (loss)
                    loss = 0.0
    return training_losses


#%matplotlib inline
#import matplotlib.pyplot as plt
x = train_network()
#plt.plot(x)





