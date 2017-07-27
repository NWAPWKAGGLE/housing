from datetime import datetime
from glob import iglob
from os import path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

debug = False


class NeuralNet:
    @classmethod
    def new(cls, model_name, shape, base_learning_rate=0.001, dtype=np.float32, weight_generator=None,
            bias_generator=None, activator=tf.sigmoid, activate_output=False,
            error_metric=tf.losses.mean_squared_error, optimizer=tf.train.GradientDescentOptimizer):

        x = tf.placeholder(dtype, (None, shape[0]), name='x')
        y_ = tf.placeholder(dtype, (None, shape[-1]), name='y_')
        y = NeuralNet._build_net(x, shape, weight_generator or cls.default_weight_generator,
                                 bias_generator or cls.default_bias_generator, activator, activate_output)

        err = tf.identity(error_metric(y_, y), name="err")
        train = optimizer(base_learning_rate, name="train").minimize(err)

        return cls(model_name)

    @classmethod
    def load(cls, model_name, save_dir=path.join('', 'model_saves')):

        selector = path.join(save_dir, model_name, '*.ckpt.meta')
        newest = max(iglob(selector), key=path.getctime)
        if debug:
            print(newest)
        return cls(model_name, newest)

    def __init__(self, model_name, restore_file=None):
        self.managed = False
        self.model_name = model_name
        self.restore_file = restore_file
        self.trained = (self.restore_file is not None)

    def __enter__(self):
        sess = tf.Session()
        if self.restore_file is not None:
            saver = tf.train.import_meta_graph(self.restore_file)
            saver.restore(sess, self.restore_file[:-5])
            self.saver = saver
            self.trained = True
        else:
            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.managed = True
        return self

    def __exit__(self, type, value, traceback):
        self.managed = False
        self.saver = None
        self.sess.close()

    @staticmethod
    def _build_net(x_placeholder, shape, weight_generator, bias_generator, activator, activate_output):
        a_set = [x_placeholder]
        for i in range(len(shape) - 1):
            up_size = shape[i]
            down_size = shape[i + 1]

            if debug:
                print('mid-layer [{0}, {1}]'.format(up_size, down_size))

            w = tf.Variable(weight_generator([up_size, down_size]))
            b = tf.Variable(bias_generator([down_size]))
            if (i == len(shape) - 2):  # output
                if activate_output:
                    a_set.append(activator(tf.matmul(a_set[-1], w) + b, name='y'))
                else:
                    a_set.append(tf.add(tf.matmul(a_set[-1], w), b, name='y'))
            else:  # upstream
                a_set.append(tf.matmul(a_set[-1], w) + b)
        return a_set[-1]

    @staticmethod
    def default_weight_generator(shape):
        return tf.truncated_normal(shape, stddev=.1)

    @staticmethod
    def default_bias_generator(shape):
        return tf.constant(0.1, shape=shape)

    @staticmethod
    def xavier_weight_generator(shape):
        raise NotImplementedError("Xavier generation isn't implemented yet")

    def learn(self, xvals, y_vals, epochs, report_interval=10000, save_dir=path.join('', 'model_saves')):
        if not self.managed:
            raise RuntimeError("Class TFRunner must be resource-managed by _with_ statement")
        if debug:
            print(self.sess.run('x:0', feed_dict={'x:0': xvals}))
            print(self.sess.run('y_:0', feed_dict={'y_:0': y_vals}))

        for i in tqdm(range(epochs)):
            self.sess.run('train', feed_dict={'x:0': xvals, 'y_:0': y_vals})
            if (i % report_interval == 0) or (i + 1 == epochs):
                measured_error = self.sess.run('err:0', feed_dict={'y_:0': y_vals, 'x:0': xvals})
                save_path = self._save(save_dir, measured_error, i, epochs)
                if debug:
                    y = self.sess.run('y:0', feed_dict={'x:0': xvals})
                    print('y(x)')
                    print(y)
                    print('y(x) measured error')
                    print(measured_error)
                    print("Model saved in file: {0}".format(save_path))
        self.trained = True

    def _save(self, save_dir, err, i, epochs):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        s_path = path.join(save_dir, self.model_name, '{0}__{1}_{2}__{3}.ckpt'.format(err, i, epochs,
                                                                                      str(datetime.now()).replace(':',
                                                                                                                  '_')))
        return self.saver.save(self.sess, s_path)

    def validate(self, qvals, q_vals):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("This TFRunner has not been trained yet")
            else:
                measured_error = self.sess.run('err:0', feed_dict={'x:0': qvals, 'y_:0': q_vals})
                if debug:
                    print("q")
                    print(self.sess.run('x:0', feed_dict={'x:0': qvals}))
                    print("y(q)")
                    print(self.sess.run('y:0', feed_dict={'x:0': qvals}))
                    print("q_")
                    print(self.sess.run('y_:0', feed_dict={'y_:0': q_vals}))
                    print("y(q) error")
                    print(measured_error)
                return measured_error

    def feed_forward(self, tvals):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("This TFRunner has not been trained yet")
            else:
                result = self.sess.run('y:0', feed_dict={'x:0': tvals})
                if debug:
                    print("t")
                    print(self.sess.run('x:0', feed_dict={'x:0': tvals}))
                    print("y(t)")
                    print(result)
                return result
