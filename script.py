import Parity_Data_Generator
import numpy as np
from newESN import ESN
import time

N = 4000   # number of datapoints
n = 3       # n-parity
timescale = 50
rng = np.random.RandomState(1578) #TODO
inputESN_reservoirSize=300
outputESN_reservoirSize= 50
input_spectral_rad = 1.3
input_leak_rate = 0.7
output_leak_rate = 1.5
output_spectral_rad = 1.8

inputESN = ESN(inputs=1,
               neurons=inputESN_reservoirSize,
               spectral_radius=input_spectral_rad,
               leak_rate= input_leak_rate,  # adjust leak_rate
               sparsity=0.95,  # 0.95
               dt=0.1,
               noise=0.01,
               input_scale=2,
               input_shift=-1)
inputESN_ident_Matrix=np.eye(inputESN_reservoirSize)

outputESN = ESN(inputs=inputESN_reservoirSize,
                neurons=outputESN_reservoirSize,
                spectral_radius=output_spectral_rad,
                leak_rate= output_leak_rate,  # adjust leak_rate
                dt= 0.1,
                sparsity=0.7,
                noise=0.01,
                input_scale=0.85,
                input_shift=0)

# Generate training data
bits, _, target = Parity_Data_Generator.generateParityData(N, n, timescale=timescale, randomstate=rng)
traintest_cutoff = int(np.ceil(0.7 * len(bits)))
train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
train_targets, test_targets = target[:traintest_cutoff], target[traintest_cutoff:]


washout=int(len(train_bits)/10)
intermediate_train = inputESN.predict(train_bits, readout_weights=inputESN_ident_Matrix)
outputESN.train(intermediate_train, train_targets, washout=washout)

intermediate_test = inputESN.predict(test_bits, readout_weights=inputESN_ident_Matrix)
final_prediction = outputESN.predict(intermediate_test)

error = np.sqrt(np.mean((final_prediction - test_targets) ** 2))
