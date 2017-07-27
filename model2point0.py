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




