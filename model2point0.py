import tensorflow as tf
from tf_runner import NeuralNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#load training features and expected outputs
training_features = np.genfromtxt("./data/trainfeaturesnew.csv", delimiter=',', dtype=None)
training_expected = np.genfromtxt("./data/trainexpected.csv", delimiter=',', dtype=None)

#load validation features and expected outputs
validation_expected = np.genfromtxt("./data/validationexpected.csv", delimiter=',', dtype=None)
validation_features = np.genfromtxt("./data/validationfeaturesnew.csv", delimiter=',', dtype=None)

#load test data
test_features = np.genfromtxt("./data/testfeaturesnew.csv", delimiter=',', dtype=None)

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

for i in range(len(test_features)):
    test_features[i][0] = test_features[i][0] / 1000
    test_features[i][1] = test_features[i][1] / 1000
    test_features[i][4] = test_features[i][4] / 1000
    test_features[i][7] = test_features[i][7] / 1000
    test_features[i][8] = test_features[i][8] / 100
    test_features[i][9] = test_features[i][9] / 100
    test_features[i][11] = test_features[i][11] / 100
    test_features[i][13] = test_features[i][13] / 100

####Hyperparams
learning_rate = .05
epochs = 100
shape = [33, 1000, 750, 550, 100, 50, 1]
activate_output = True
report_interval = 1000
model_name = "model2point0point8"


training_errors = []
validation_errors = []


with NeuralNet.new(model_name, shape=shape, activate_output=activate_output, optimizer_params=learning_rate) as NN:
    print("starting model")
    for i in range(100):
        NN.learn(training_features, training_expected, epochs, report_interval=report_interval)
        training_errors.append(NN.learn(training_features, training_expected, epochs, report_interval=report_interval))
        validation_errors.append(NN.validate(validation_features, validation_expected))
        validation_output = NN.feed_forward(validation_features)
        plt.clf()
        plt.plot(1000000*validation_output)
        plt.plot(1000000*validation_expected)
        plt.savefig('./graphs/convergence{0}'.format(i), bbox_inches='tight')

    plt.clf()
    plt.plot(training_errors, label='training error')
    plt.plot(validation_errors, label='validation error')
    plt.show()

    np.savetxt("./data/testoutputs.csv", 1000000*NN.feed_forward(test_features), delimiter=",", fmt="%f")



