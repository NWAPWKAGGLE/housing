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
    @classmethod
    def new(cls, model_name, shape, dtype=np.float32):
        x = tf.placeholder(dtype, (None, shape[0]), name='x')
        y_ = tf.placeholder(dtype, (None, shape[-1]), name='y_')

    @classmethod
    def load(cls, filename):
        saver = tf.train.import_meta_graph(filename + ".meta")
        saver.restore(self.sess, filename)
        graph = tf.get_default_graph()
        self.y = graph.get_tensor_by_name('y:0')
        self.x = graph.get_tensor_by_name('x:0')
        self.y_ = graph.get_tensor_by_name('y_:0')
        self.trained = True

    def __init__(self, model_name, shape, save_dir):
        self.managed = False
        self.trained = False
        if len(shape) < 2:
            raise ValueError("Shape must be at least 2 in length")
        self.shape = shape
        self.x_placeholder = tf.placeholder(tf.float32, (None, shape[0]), name='x')
        self.y__placeholder = tf.placeholder(tf.float32, (None, shape[-1]), name='y_')
        self.model_name = model_name
        #self.dtype = dtype
        self.save_dir = path.join(save_dir, model_name)

    def __enter__(self):
        self.managed = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self):
        self.managed = False
        self.sess.close()

    @staticmethod
    def _build_net(x_placeholder, shape, weights_generator, bias_generator, activator, activate_output):
        a_set = [x_placeholder]
        for i in range(len(shape) - 1):
            up_size = shape[i]
            down_size = shape[i + 1]

            print('on layer with up {0} and down {1}'.format(up_size, down_size))

            w = tf.Variable(weights_generator([up_size, down_size]))
            b = tf.Variable(bias_generator([down_size]))
            if (i < len(shape) - 2):  # output
                if activate_output:
                    a_set.append(activator(tf.matmul(a_set[-1], w) + b, name='y'))
                else:
                    a_set.append(tf.add(tf.matmul(a_set[-1], w), b, name='y'))
            else:  # upstream
                a_set.append(tf.matmul(a_set[-1], w) + b)
        return a_set[-1]

    def learn(self, xvals, y_vals, epochs, learning_rate, report_interval=10000,
              qvals=None, q_vals=None, error_metric=tf.losses.mean_squared_error,
              optimizer=tf.train.GradientDescentOptimizer, activator=tf.sigmoid,
              activate_output=True, weights_generator=initialize_weights,
              bias_generator=initialize_biases):
        if not self.managed:
            raise RuntimeError("Class TFRunner must be resource-managed by _with_ statement")
        if not self.trained:
            self.y = TFRunner._build_net(self.x, self.shape, weights_generator, bias_generator, activator,
                                         activate_output)

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
                    print(self.sess.run('x', feed_dict={'x': qvals}))
                    print("y(q)")
                    print(self.sess.run('y', feed_dict={'x': qvals}))
                    print("q_")
                    print(self.sess.run('y_', feed_dict={'y_': q_vals}))
                    print("y(q) error")
                    print(self.sess.run(err, feed_dict={'x': qvals, 'y_': q_vals}))

                save_path = path.join(self.save_dir,
                                      '{1}__{2}.ckpt'.format(measured_error, str(datetime.now()).replace(':', '_')))
                save_path = saver.save(self.sess, save_path)
                print("Model saved in file: {0}".format(save_path))

        self.trained = True

    def run(self, vals, q_vals=None):
        if not self.trained:
            raise RuntimeError("Model is untrained")
        else:
            return self.sess.run(self.y, feed_dict={self.x: vals})


if __name__ == '__main__':
    with TFRunner('tftest', [1, 10, 1]) as runner:
        pass
