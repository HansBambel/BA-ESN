import Parity_Data_Generator
import InputToParityESN
import ParityToOutputESN
import numpy as np

rng = np.random.RandomState(42)

N = 20000   # number of datapoints
n = 3       # n-parity
zero = [1, 0, 1, 0, 1, 0]      # encoding for a zero
one =  [1, 1, 1, 0, 0, 0]      # encoding for a one
# produce Data
bits, parity, target = Parity_Data_Generator.generateParityData(N, n, zero, one, randomstate=rng)

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

##################################
print("### Parity-->Target ###")

#prepare Data
# predicted Parity-size is only a portion of targets --> multiply each by length of encoding
traintest_cutoff_targets = int(np.ceil(0.7*len(target)))
train_targets, test_targets = target[:traintest_cutoff_targets], target[traintest_cutoff_targets:]

targetESN = ParityToOutputESN.fastESN(rng)
targetESN.fit(predictedParity, train_targets)
predictedTargets = targetESN.predict(test_targets)
print("Test error")
print(np.sqrt(np.mean((predictedTargets-test_targets)**2)))
