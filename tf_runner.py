from datetime import datetime
from glob import iglob
from os import path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

debug = False


class NeuralNet:
    """
        A deep neural net, implemented on top of TensorFlow. Supports saving and loading. Debug info can be output by
        setting debug = True in the module namespace.
    """


    @classmethod
    def new(cls, model_name, shape, optimizer_params, dtype=np.float32, weight_generator=None,
            bias_generator=None, activator=tf.sigmoid, activate_output=False,
            error_metric=tf.losses.mean_squared_error, optimizer=tf.train.GradientDescentOptimizer):
        """
            Create a new deep neural net from scratch.
        :param model_name: The folder name to use for model saves.
        :param shape: A list of integers of the form [input, deep1, deep2, deepn, output] which represents
            the number of neurons in each layer. The minimum number of layers is 2.
        :param optimizer_params: A dict of parameters (or a single parameter) to pass to the optimizer's
        __init__ method. When using the default tf.train.GradientDescentOptimizer, passing in a single value sets the
        learning rate.
        :param dtype: The datatype of input data - np.[dtype]. Defaults to np.float32.
        :param weight_generator: A function which takes a single list parameter of form [upstream_layer_size,
        downstream_layer_size] and returns a rank-2 tf.Tensor of initialized weights. Defaults to a tensor filled with
        tf.truncated_normal(stddev=0.1).
        :param bias_generator: A function which takes a single list parameter of form [downstream_layer_size] and
        returns a rank-1 tf.Tensor of initialized biases. Defaults to a tensor filled with 0.1.
        :param activator: Activation function. A function handle which takes in a single float parameter z and returns
        the output of the function applied to z. Defaults to tf.sigmoid.
        :param activate_output: Whether the outputs from the output layer should have the activation function applied.
        Defaults to False.
        :param error_metric: Error metric. A function handle which takes parameters (predicted, actual) and returns a
        metric of the error. Defaults to tf.losses.mean_squared_error.
        :param optimizer: Optimization algorithm. A function handle which takes parameters specified by optimizer_params
        and returns an object with a minimize method into which the error metric is passed. Defaults to
        tf.train.GradientDescentOptimizer.
        :return: An untrained instance of NeuralNet on which learn() may be called.
        :raises ValueError: if the number of layers is less than 2.
        """

        if (len(shape) < 2):
            raise ValueError("Number of layers must be greater than 2")
        x = tf.placeholder(dtype, (None, shape[0]), name='x')
        y_ = tf.placeholder(dtype, (None, shape[-1]), name='y_')
        y = NeuralNet._build_net(x, shape, weight_generator or cls.default_weight_generator,
                                 bias_generator or cls.default_bias_generator, activator, activate_output)

        err = tf.identity(error_metric(y_, y), name='err')
        train = None
        if type(optimizer_params) is dict:
            train = optimizer(**optimizer_params, name='train').minimize(err)
        else:
            train = optimizer(optimizer_params, name='train').minimize(err)

        return cls(model_name)

    @classmethod
    def load(cls, model_name, file_name=None, save_dir=path.join('', 'model_saves')):
        """
            Load a previously saved neural net for a particular model. Behavior is undefined if the model was not
            output by an instance of NeuralNet.
        :param model_name: The model name.
        :param file_name: The relative or absolute path to the .ckpt.meta file for the particular save file
        collection. If unspecified, the most recent save for the model is used.
        :param save_dir: The relative or absolute path to the directory where the model directory is located.
        If unspecified, defaults to './model_saves'.'.
        :return: A trained instance of NeuralNet.
        """
        if file_name is None:
            selector = path.join(save_dir, model_name, '*.ckpt.meta')
            newest = max(iglob(selector), key=path.getctime)
            if debug:
                print(newest)
            file_name = newest
        return cls(model_name, file_name)

    def __init__(self, model_name, restore_file=None):
        """
        ***PRIVATE CONSTRUCTOR. DO NOT INSTANTIATE DIRECTLY. USE NeuralNet.new or NeuralNet.load.***
        Stores model information and possible file location in preparation for creating session with __enter__.
        :param model_name: the model name
        :param restore_file: a restore file path if the model should be restored; defaults to None.
        """
        self.managed = False
        self.model_name = model_name
        self.restore_file = restore_file
        self.trained = (self.restore_file is not None)

    def __enter__(self):
        """
        Enter a managed-resource block (with block). Start (and populate, if necessary) a tf.Session and tf.Saver.
        :return: self
        """
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
        """
        Exit a managed-resource block (with block). End the tf.Session and tf.Saver.
        :param type: exception type
        :param value: exception value
        :param traceback: exception traceback
        """
        self.managed = False
        self.saver = None
        self.sess.close()

    @staticmethod
    def _build_net(x_placeholder, shape, weight_generator, bias_generator, activator, activate_output):
        """
        Build a forward-fed deep net of Tensors with the specified x placeholder handle, shape, generators, and
        activator. If activate_output is False, the activator will not be applied on the outputs of the output layer.
        :param x_placeholder: A tf.placeholder which will eventually be populated with input values.
        :param shape: A list of integers of the form [input, deep1, deep2, deepn, output] which represents
            the number of neurons in each layer.
        :param weight_generator: A function which takes a single list parameter of form [upstream_layer_size,
        downstream_layer_size] and returns a rank-2 Tensor of initialized weights.
        :param bias_generator: A function which takes a single list parameter of form [downstream_layer_size] and
        returns a rank-1 Tensor of initialized biases.
        :param activator: Activation function. A function handle which takes in a single float parameter z and returns
        the output of the function applied to z.
        :param activate_output: Whether the outputs from the output layer should have the activation function applied.
        :return: a tf.Tensor handle which will eventually hold the outputs.
        """
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
        """
        Default weight initalizer. Uses tf.truncated_normal.
        :param shape: A list of form [upstream_layer_size, downstream_layer_size]
        :return: A rank-2 tf.Tensor representing the weights.
        """
        return tf.truncated_normal(shape, stddev=.1)

    @staticmethod
    def default_bias_generator(shape):
        """
        Default bias initializer. Uses 0.1.
        :param shape: A list of form [downstream_layer_size]
        :return: A rank-1 tf.Tensor representing the biases.
        """
        return tf.constant(0.1, shape=shape)

    def learn(self, xvals, y_vals, epochs, report_interval=10000, save_dir=path.join('', 'model_saves')):
        """
        Optimize the weights and biases to match xvals with y_vals as closely as possible. Optimization is performed
        epochs times and save files are generated every report_interval epochs.
        :param xvals: A numpy-array-like set of x values.
        :param y_vals: A numpy-array-like set of y_ values.
        :param epochs: The number of epochs to train for.
        :param report_interval: The epoch interval at which to generate save files.
        :param save_dir: Optional save directory. Defaults to './model_saves'.
        :raises RuntimeError: if the TFRunner instance is not resource managed.
        """
        if not self.managed:
            raise RuntimeError("Class TFRunner must be resource-managed by _with_ statement")
        if debug:
            print(self.sess.run('x:0', feed_dict={'x:0': xvals}))
            print(self.sess.run('y_:0', feed_dict={'y_:0': y_vals}))

        epoch_iter = tqdm(range(epochs)) if debug else range(epochs)
        for i in epoch_iter:
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
        """
        Write out the current state to a checkpoint file in save_dir. The name is formatted as 'err__i_epochs__datetime'
        where err is the current error as measured by error_metric, i is the current epoch, and epochs is the total
        number of epochs.
        :param save_dir: The location to save the files.
        :param err: The current error, as measured by error_metric.
        :param i: The current epoch.
        :param epochs: The total number of epochs to be executed in this learn() call.
        :return: The path to the saved file.
        :raises RuntimeError: if the TFRunner instance is not resource managed.
        """
        if not self.managed:
            raise RuntimeError("TFRunner must be in with statement")
        s_path = path.join(save_dir, self.model_name, '{0}__{1}_{2}__{3}.ckpt'.format(err, i, epochs,
                                                                                      str(datetime.now()).replace(':',
                                                                                                                  '_')))
        return self.saver.save(self.sess, s_path)

    def validate(self, qvals, q_vals):
        """
        Feed forward a set of validation data. Return the total error according to the configured error_metric.
        :param qvals: The inputs.
        :param q_vals: The expected outputs.
        :return: The error according to error_metric.
        :raises RuntimeError: if the TFRunner instance is not resource managed or is untrained.
        """
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
        """
        Feed forward a set of data and return the outputs directly.
        :param tvals: the inputs
        :return: the outputs
        :raises RuntimeError: if the TFRunner instance is not resource managed or is untrained.
        """
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
