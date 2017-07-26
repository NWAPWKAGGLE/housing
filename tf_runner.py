import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Process, Pipe
from os import path


def initialize_weights(shape):
    return tf.truncated_normal(shape, stddev=.1)


def initialize_biases(shape):
    return tf.constant(0.1, shape=shape)


# TODO: Refactor generation of variables and placeholders to be accessible by name

class TFRunner:
    def __init__(self, model_name, shape, dtype=np.float32, save_dir='./model_saves/'):
        self.managed = False
        self.trained = False
        if len(shape) < 2:
            raise ValueError("Shape must be at least 2 in length")

        self.shape = shape
        self.x = tf.placeholder(dtype, (None, shape[0]))
        self.y_ = tf.placeholder(dtype, (None, shape[-1]))
        self.y = None
        self.model_name = model_name
        self.dtype = dtype
        self.save_dir = path.join(save_dir, model_name)

    def __enter__(self):
        self.managed = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self):
        self.managed = False
        self.sess.close()

    def learn(self, xvals, y_vals, epochs, learning_rate, report_interval=10000,
              qvals=None, q_vals=None, error_metric=tf.losses.mean_squared_error,
              optimizer=tf.train.GradientDescentOptimizer, activator=tf.sigmoid,
              activate_output=True, weights_generator=initialize_weights,
              bias_generator=initialize_biases):
        if not self.managed:
            raise RuntimeError("Class TFRunner must be resource-managed by _with_ statement")
        if not self.trained:
            a_set = [self.x]
            w_set = []
            b_set = []
            for i in range(len(self.shape) - 1):
                up_size = self.shape[i]
                down_size = self.shape[i + 1]

                print('on layer with up {0} and down {1}'.format(up_size, down_size))

                w = tf.Variable(weights_generator([up_size, down_size]))
                b = tf.Variable(bias_generator([down_size]))
                w_set.append(w)
                b_set.append(b)
                if (i < len(self.shape) - 2) or activate_output:
                    a_set.append(activator(tf.matmul(a_set[-1], w) + b))
                else:
                    a_set.append(tf.matmul(a_set[-1], w) + b)
            self.y = a_set[-1]
        else:
            ...
        err = error_metric(self.y_, self.y)
        train = optimizer(learning_rate).minimize(err)

        print(self.sess.run(self.x, feed_dict={self.x: xvals}))
        print(self.sess.run(self.y_, feed_dict={self.y_: y_vals}))

        saver = tf.train.Saver()

        for i in tqdm(range(epochs)):
            self.sess.run(train, feed_dict={self.x: xvals, self.y_: y_vals})
            if (i % report_interval == 0) or (i + 1 == epochs):
                measured_error = self.sess.run(err, feed_dict={self.x: xvals, self.y_: y_vals})

                print('y(x)')
                print(self.sess.run(self.y, feed_dict={self.x: xvals}))
                print("y(x) error")
                print(measured_error)

                if (qvals is not None) and (q_vals is not None):
                    print('q')
                    print(self.sess.run(x, feed_dict={x: qvals}))
                    print("y(q)")
                    print(self.sess.run(y, feed_dict={x: qvals}))
                    print("q_")
                    print(self.sess.run(y_, feed_dict={y_: q_vals}))
                    print("y(q) error")
                    print(self.sess.run(err, feed_dict={x: qvals, y_: q_vals}))

                save_path = path.join(self.save_dir,
                                      '{1}__{2}.ckpt'.format(measured_error, str(datetime.now()).replace(':', '_')))
                save_path = saver.save(self.sess, save_path)
                print("Model saved in file: {0}".format(save_path))

        self.trained = True

    def load(self, filename=None):
        saver = tf.train.Saver()
        if filename is not None:
            saver.restore(self.sess, filename)
            self.trained = True
        else:
            raise NotImplementedError("Most recent save open isn't implemented yet")
