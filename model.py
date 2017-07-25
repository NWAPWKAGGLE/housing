import tensorflow as tf
from tqdm import tqdm
import numpy as np

##these functions initialize weights and biases with non-zero values
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def rmsle(predicted, real, length):
    sum=0.0

    p = tf.log(predicted*1000000+1)
    r = tf.log(real*1000000+1)
    sum = tf.reduce_sum(p - r)**2
    return (sum/length)**0.5

#### HYPERPARAMETERS

learning_rate = .001
num_epochs = 10000
num_training_inputs = 1460
#list of training files
filename_queue = tf.train.string_input_producer(["./data/5features.csv"])
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
W1 = weight_variable([5, 15])
b1 = bias_variable([15])

## output of first hidden layer
a1 = tf.sigmoid(tf.matmul(x, W1) + b1)

## weights and biases for output layer - 1 neuron
W2 = weight_variable([15, 1])
b2 = bias_variable([1])

# output of second layer/model
y = tf.tanh(tf.matmul(a1, W2)+b2)

# expected output
y_ = tf.placeholder(tf.float32)


# root mean squared logarithmic error

rmslerror = rmsle(y, y_, num_training_inputs)

# mean squared error
error = tf.losses.mean_squared_error(y_, y)

# gradient descent for learning
trainstep = tf.train.AdamOptimizer(learning_rate).minimize(error)

with tf.Session() as sess:
    # Populate filename queue
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #initialize the variables and graph
    init = tf.global_variables_initializer()
    sess.run(init)

    #load in data
    inputs = []
    expected_outputs = []

    #loop through by line
    for i in tqdm(range(num_training_inputs)):

        #grab line value
        feature, expected_output = sess.run([features, col6])

        #add line to inputs/expected outputs
        inputs.append(feature)
        expected_outputs.append(expected_output/1000000)
    coord.request_stop()
    coord.join(threads)
    inputs = np.resize(inputs, (num_training_inputs, 5))

    expected_outputs = np.resize(expected_outputs, (num_training_inputs, 1))

    for j in tqdm(range(num_epochs)):
        sess.run(trainstep, feed_dict = {x: inputs, y_: expected_outputs})

    print(sess.run(rmslerror, feed_dict = {x: inputs, y_: expected_outputs}))

