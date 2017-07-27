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

#scale outputs
training_expected = training_expected/1000000
validation_expected = validation_expected/1000000

#scale features
for i in range(len(training_features)):
    training_features[i][0] = training_features[i][0] / 1000
    training_features[i][1] = training_features[i][1] / 1000
    training_features[i][4] = training_features[i][4] / 1000
    training_features[i][7] = training_features[i][7] / 1000
    training_features[i][8] = training_features[i][8] / 100
    training_features[i][9] = training_features[i][9] / 100
    training_features[i][11] = training_features[i][11] / 100
    training_features[i][13] = training_features[i][13] / 100

for i in range(len(validation_features)):
    validation_features[i][0] = validation_features[i][0] / 1000
    validation_features[i][1] = validation_features[i][1] / 1000
    validation_features[i][4] = validation_features[i][4] / 1000
    validation_features[i][7] = validation_features[i][7] / 1000
    validation_features[i][8] = validation_features[i][8] / 100
    validation_features[i][9] = validation_features[i][9] / 100
    validation_features[i][11] = validation_features[i][11] / 100
    validation_features[i][13] = validation_features[i][13] / 100


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





