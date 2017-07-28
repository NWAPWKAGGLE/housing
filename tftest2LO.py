from tf_runner import NeuralNet
import numpy as np

learning_rate = 0.001
epochs = 100000
shape = [1, 10, 1]
activate_output = False
report_interval = 10000
model_name = 'tftest2L4'

seed = np.arange(1, 15, dtype=np.int32)
xvals = np.reshape(seed, (len(seed), 1))
y_vals = xvals + 4
qvals = xvals + 2
q_vals = qvals + 4
tvals = xvals - 3

with NeuralNet.load(model_name) as net:
    # net.learn(xvals, y_vals, epochs, report_interval=report_interval, progress_bar=True)
    print(net.validate(qvals, q_vals))
    print(net.feed_forward(tvals))
