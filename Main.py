import numpy as np

import InputToParityESN
import ParityToOutputESN
import Parity_Data_Generator

rng = np.random.RandomState(42)

N = 20000   # number of datapoints
n = 3       # n-parity
# produce Data
bits, parity, target = Parity_Data_Generator.generateParityData(N, n, randomstate=rng)

# Divide in training and test data
traintest_cutoff = int(np.ceil(0.7 * len(bits)))
train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
train_parity, test_parity = parity[:traintest_cutoff], parity[traintest_cutoff:]
train_targets, test_targets = target[:traintest_cutoff], target[traintest_cutoff:]

print("### Input-->Parity ###")
# get good configs for a slow ESN
nParityESN = InputToParityESN.slowESN(n, rng)

nParityESN.fit(train_bits, train_parity)
predictedParity = nParityESN.predict(test_bits)
print("Testing error")
print(np.sqrt(np.mean((predictedParity - test_parity) ** 2)))

##################################
print("### Parity-->Target ###")

targetESN = ParityToOutputESN.fastESN(rng)
targetESN.fit(predictedParity, train_targets)
predictedTargets = targetESN.predict(test_targets)
print("Test error")
print(np.sqrt(np.mean((predictedTargets-test_targets)**2)))
