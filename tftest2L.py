from jrstnets import DFFNetFactory, WeightBiasGenerators
import numpy as np

learning_rate = 0.001
epochs = 10000
shape = [1, 10, 1]
activate_output = False
report_interval = 10000
model_name = 'tftest2M1'

seed = np.arange(1, 15, dtype=np.int32)
xvals = np.reshape(seed, (len(seed), 1))
y_vals = xvals + 4
qvals = xvals + 2
q_vals = qvals + 4
tvals = xvals - 3

with DFFNetFactory.load_or_new(model_name, shape, optimizer_params=learning_rate, activate_output=False,
                               weight_generator=WeightBiasGenerators.default_weight_generator) as net:
    if not net.trained:
        print('training')
        net.learn(xvals, y_vals, epochs, report_interval=report_interval, progress_bar=True)
    else:
        print('pretrained')
        net.learn(xvals, y_vals, 100, report_interval=1000, progress_bar=True)
    print(net.validate(qvals, q_vals))
    print(net.feed_forward(tvals))
