import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime

############ TUNABLE ################
learning_rate = 0.25
epochs = 1000000
hidden_shape = [10]
activate = tf.sigmoid
activate_output = False
error_metric = tf.losses.mean_squared_error
optimizer = tf.train.GradientDescentOptimizer
report_interval = 10000
model_name = 'tftest'


def initalize_weights(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def initialize_biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


seed = np.arange(1, 11, dtype=np.int32)
xvals = np.reshape(seed, (len(seed), 1))
yvals = xvals + 4
qvals = xvals + 2
q_vals = qvals + 4

x = tf.placeholder(np.float32, (None, 1))
y_ = tf.placeholder(np.float32, (None, 1))
############### END TUNABLE ################

a_set = [x]
full_shape = [1] + hidden_shape + [1]
for i in range(len(full_shape) - 1):
    up_size = full_shape[i]
    down_size = full_shape[i + 1]

    print('on layer with up {0} and down {1}'.format(up_size, down_size))

    w = initalize_weights([up_size, down_size])
    b = initialize_biases([down_size])
    if (i < len(full_shape) - 2) or activate_output:
        a_set.append(activate(tf.matmul(a_set[-1], w) + b))
    else:
        a_set.append(tf.matmul(a_set[-1], w) + b)
y = a_set[-1]

# w1 = initalize_weights([1, 10])
# b1 = initialize_biases([10])
# a1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# w2 = initalize_weights([10, 1])
# b2 = initialize_biases([1])
# a2 = tf.matmul(a1, w2) + b2
# y = a2

err = error_metric(y_, y)
train = optimizer(learning_rate).minimize(err)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(x, feed_dict={x: xvals}))
    print(sess.run(y_, feed_dict={y_: yvals}))

    for i in tqdm(range(epochs)):
        sess.run(train, feed_dict={x: xvals, y_: yvals})
        if i % report_interval == 0:
            print('y(x)')
            print(sess.run(y, feed_dict={x: xvals}))

            print('y(x) error')
            measured_error = sess.run(err, feed_dict={x: xvals, y_: yvals})
            print(measured_error)

            print('q')
            print(sess.run(x, feed_dict={x: qvals}))
            print('y(q)')
            print(sess.run(y, feed_dict={x: qvals}))
            print('q_')
            print(sess.run(y_, feed_dict={y_: q_vals}))
            print('y(q) error')
            print(sess.run(err, feed_dict={x: qvals, y_: q_vals}))

            save_path = "./savedmodels/{0}/{1}__{2}.ckpt".format(
                model_name, measured_error, str(datetime.now()).replace(':', '_'))
            save_path = saver.save(sess, save_path)
            print("Model saved in file: {0}".format(save_path))




# expected_outputs = tf.constant([x**2+x+4 for x in seed], shape=[len(seed), 1])
# dot = tf.matmul(a, b)
# dyadic = tf.matmul(b, a)

# with tf.Session() as sess:
#    print(a.eval())
#    print(b.eval())
#    print(sess.run(dot))
#    print(sess.run(dyadic))
