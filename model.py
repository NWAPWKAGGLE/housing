import tensorflow as tf


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
filename_queue = tf.train.string_input_producer(["data/5features.csv"])
#read files
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

#default values and type of input
record_defaults = [[9000], [1989], [3], [6], [2007]]

#read columns
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

#### MODEL

# this is the placeholder for the inputs. takes an arbitrary size of training set with 5 features in each example
x = tf.placeholder(tf.float32, (None, 5))

## weights and biases for the first layer. first layer has 10 neurons
W1 = weight_variable([5, 10])
b1 = bias_variable([10])

##output of first layer
a1 = tf.sigmoid(tf.matmul(W1, x) + b1)

## weights and biases for second/output layer. has one output
W2 = weight_variable([10, 1])
b2 = bias_variable([1])

# output of second layer/model
y = tf.matmul(W2, a1) + b2

# expected output
y_ = tf.placeholder(tf.float32, 'y hat')

# mean squared error
error = tf.losses.mean_squared_error(y_, y)

# gradient descent for learning
trainstep = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)


