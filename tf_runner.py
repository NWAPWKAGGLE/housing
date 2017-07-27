from datetime import datetime
from glob import iglob
from os import path

import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
        learning_rate = tf.Variable(base_learning_rate, dtype=tf.float32, name="learning_rate")
        train = optimizer(learning_rate, name="train").minimize(err)

        return cls(model_name)

    @classmethod
    def load(cls, model_name, save_dir=path.join('', 'model_saves')):

        selector = path.join(save_dir, model_name, '*.ckpt.meta')
        newest = max(iglob(selector), key=path.getctime)

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
            self.trained = True
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        self.managed = True
        return self

    def __exit__(self):
        self.managed = False
        self.sess.close()

    @staticmethod
    def _build_net(x_placeholder, shape, weight_generator, bias_generator, activator, activate_output):
        a_set = [x_placeholder]
        for i in range(len(shape) - 1):
            up_size = shape[i]
            down_size = shape[i + 1]

            print('on layer with up {0} and down {1}'.format(up_size, down_size))

            w = tf.Variable(weight_generator([up_size, down_size]))
            b = tf.Variable(bias_generator([down_size]))
            if (i < len(shape) - 2):  # output
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

    def learn(self, xvals, y_vals, epochs, report_interval=10000, learning_rate=None,
              save_dir=path.join('', 'model_saves')):

        if not self.managed:
            raise RuntimeError("Class TFRunner must be resource-managed by _with_ statement")

        if learning_rate is not None:
            self.sess.run(tf.assign([v for v in tf.global_variables() if v.op.name == 'learning_rate'][0],
                                    learning_rate))

        print(self.sess.run('x', feed_dict={'x': xvals}))
        print(self.sess.run('y_', feed_dict={'y_': y_vals}))

        saver = tf.train.Saver()

        for i in tqdm(range(epochs)):
            self.sess.run('train', feed_dict={'x': xvals, 'y_': y_vals})
            if (i % report_interval == 0) or (i + 1 == epochs):
                measured_error = self.sess.run('err', feed_dict={'x': xvals, 'y_': y_vals})

                print('y(x)')
                print(self.sess.run('y', feed_dict={'x': xvals}))
                print("y(x) error")
                print(measured_error)

                save_path = path.join(save_dir,
                                      '{0}__{1}.ckpt'.format(measured_error, str(datetime.now()).replace(':', '_')))
                save_path = saver.save(self.sess, save_path)
                print("Model saved in file: {0}".format(save_path))

        self.trained = True

    def validate(self, qvals, q_vals):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("This TFRunner has not been trained yet")
            else:
                print("q")
                print(self.sess.run('x', feed_dict={'x': qvals}))
                print("y(q)")
                print(self.sess.run('y', feed_dict={'x': qvals}))
                print("q_")
                print(self.sess.run('y_', feed_dict={'y_': q_vals}))
                print("y(q) error")
                print(self.sess.run('err', feed_dict={'x': qvals, 'y_': q_vals}))

    def run(self, tvals):
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        else:
            if not self.trained:
                raise RuntimeError("This TFRunner has not been trained yet")
            else:
                result = self.sess.run('y', feed_dict={'x': tvals})
                print("t")
                print(self.sess.run('x', feed_dict={'x': tvals}))
                print("y(t)")
                print(result)
                return result
