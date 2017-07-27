from tf_runner import NeuralNet
import numpy as np

############ TUNABLE ################
learning_rate = 0.01
epochs = 1000
shape = [1, 10, 1]
activate_output = False
report_interval = 100
model_name = 'tftest2'

seed = np.arange(1, 11, dtype=np.int32)
xvals = np.reshape(seed, (len(seed), 1))
y_vals = xvals + 4
qvals = xvals + 2
q_vals = qvals + 4
tvals = xvals - 3

with NeuralNet.new('tftest3', [1, 10, 1], base_learning_rate=learning_rate, activate_output=False) as net:
    net.learn(xvals, y_vals, epochs, report_interval=report_interval)
    net.validate(qvals, q_vals)
    net.feed_forward(tvals)

print('net exited')

with NeuralNet.load('tftest3') as net:
    net.feed_forward(tvals)
