import Parity_Data_Generator
import InputToParityESN
import ParityToOutputESN
import numpy as np

rng = np.random.RandomState(42)

N = 20000   # number of datapoints
n = 3       # n-parity
# produce Data
bits, parity = Parity_Data_Generator.generateParityData(N, n, randomstate=rng)

# Divide in training and test data
traintest_cutoff = int(np.ceil(0.7 * N))
train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
train_output, test_output = parity[:traintest_cutoff], parity[traintest_cutoff:]

print("### Input-->Parity ###")
# get good configs for a slow ESN
nParityESN = InputToParityESN.slowESN(n, rng)

nParityESN.fit(train_bits, train_output)
predictedParity = nParityESN.predict(test_bits)
print("Testing error")
print(np.sqrt(np.mean((predictedParity-test_output)**2)))
# predicted_Parity = InputToParityESN.fit(training_bits, training_parity, rng)


