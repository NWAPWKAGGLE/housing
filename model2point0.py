import tensorflow as tf
from tf_runner import NeuralNet
from tqdm import tqdm
import numpy as np

#load training features and expected outputs
training_features = np.genfromtxt("../data/trainfeatures.csv", delimiter=',', dtype=None)
training_expected = np.genfromtxt("../data/trainexpected.csv", delimiter=',', dtype=None)

#load validation features and expected outputs
validation_expected = np.genfromtxt("../data/validationexpected.csv", delimiter=',', dtype=None)
validation_features = np.genfromtxt("../data/validationfeatures.csv", delimiter=',', dtype=None)

#delete extra column from expected outputs
training_expected = np.delete(training_expected, 1, axis=1)
validation_expected = np.delete(validation_expected, 1, axis=1)


####Hyperparams
learning_rate = .05
epochs = 10000
shape = [57, 100, 70, 50, 30, 20, 10, 1]
activate_output = True
report_interval = 1000
model_name = "model2point0"

with NeuralNet.new(model_name, shape, optimizer_params=learning_rate, activate_output=True) as NN:
    NN.learn(training_features, training_expected, epochs, report_interval=report_interval)
    NN.validate(validation_features, validation_expected)





