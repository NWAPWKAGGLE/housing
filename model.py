import tensorflow as tf

import numpy as np


##these functions initialize weights and biases with non-zero values
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#### HYPERPARAMETERS

learning_rate = .05
num_epochs = 200

#list of training files
filename_queue = tf.train.string_input_producer(["/Users/eliwinkelman/housing/data/5features.csv"])
#read files
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#default values and type of input
record_defaults = [[9000], [1989], [3], [6], [2007], [150000]]

#read columns
col1, col2, col3, col4, col5, col6= tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5])


#### MODEL

# this is the placeholder for the input layer. takes an arbitrary size of training set with 5 features in each example
x = tf.placeholder(tf.float32, (None, 5))

## weights and biases for the first hidden layer - 10 neurons
W1 = weight_variable([5, 10])
b1 = bias_variable([10])

## output of first hidden layer
a1 = tf.sigmoid(tf.matmul(x, W1) + b1)


## weights and biases for output layer - 1 neuron
W2 = weight_variable([10, 1])
b2 = bias_variable([1])

# output of second layer/model
y = tf.matmul(a1, W2)+b2

# expected output
y_ = tf.placeholder(tf.float32)

# mean squared error
error = tf.losses.mean_squared_error(y_, y)

# gradient descent for learning
trainstep = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

with tf.Session() as sess:
    # Populate filename queue
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    init = tf.global_variables_initializer()
    sess.run(init)

    data = np.empty((1, 5))
    labels = np.array([])
    for i in range(1460):

        example, label = sess.run([features, col6])

        data = np.concatenate((data, example))
        labels = np.append(labels, label, axis=1)

    print (data)

    for j in range(num_epochs):
        sess.run(trainstep, feed_dict = {x: data, y_: labels})

    print(sess.run(error, feed_dict={x: data}))
    coord.request_stop()
    coord.join(threads)
