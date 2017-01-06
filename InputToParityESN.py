import Parity_Data_Generator
import numpy as np
from pyESN import ESN

rng = np.random.RandomState(42)
N = 20000
bits, parity = Parity_Data_Generator.generateParityData(N, n=3, randomstate=rng)

traintest_cutoff = int(np.ceil(0.7*N))

train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
train_output, test_output = parity[:traintest_cutoff], parity[traintest_cutoff:]

'''
    good parameters for n=3:
        reservoir = 500
        spectral_radius = >0.75 (from 0.9 test error is 0.0)

    n=4:
        reservoir = 1000
        spectral_radius = 0.95
'''

slowESN = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 400,        # from 200 onwards test error decreases significantly
          spectral_radius = 0.95,    # from 0.9 the test error is 0.0
          sparsity = 0.95,
          noise = 0.001,
          input_shift = 0,
          input_scaling = 3,        # 3 seems to be fine. Maybe use lower
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)

pred_train = slowESN.fit(train_bits, train_output)

print("Test error:")
pred_test = slowESN.predict(test_bits)
better_pred_test = []
for x in pred_test:
    better_pred_test.append(0 if x < 0.5 else 1)
np.array(better_pred_test)
# print(better_pred_test)
print(np.sqrt(np.mean((better_pred_test-test_output)**2)))