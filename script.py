import Parity_Data_Generator
import numpy as np
from newESN import ESN
import pickle
import os
import itertools
import time

# get Parameter according to JobID
def generateParams(ID):
    all = itertools.product(np.linspace(0.1, 1.5, num=10),
                            np.linspace(0.1, 2.0, num=10),
                            np.linspace(0.1, 1.5, num=10),
                            np.linspace(0.1, 2.0, num=10))
    # helper recipe nth
    # basically: cut n-1 from the front and take the next
    values = next(itertools.islice(all,ID, None))
    keys = "input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"
    return dict(zip(keys,values))

start_time = time.time()
# get JobID
jobID = int(os.environ.get("SGE_TASK_ID"))-1

params = generateParams(jobID)

N = 4000   # number of datapoints
n = 3       # n-parity
timescale = 50
rng = np.random.RandomState(1578) # maybe other seed
inputESN_reservoirSize=300
outputESN_reservoirSize= 50

averages = 20

errors = np.empty((averages,))
for trial in range(averages):

    inputESN = ESN(inputs=1,
                   neurons=inputESN_reservoirSize,
                   spectral_radius=params["input_spectral_rad"],
                   leak_rate= params["input_leak_rate"],  # adjust leak_rate
                   sparsity=0.95,  # 0.95
                   dt=0.1,
                   noise=0.01,
                   input_scale=2,
                   input_shift=-1)
    # for randomprojectionmatrix
    # inputESN_ident_Matrix=np.eye(inputESN_reservoirSize)

    # Generate training data
    bits, parity, _ = Parity_Data_Generator.generateParityData(N, n, timescale=timescale, randomstate=rng)
    traintest_cutoff = int(np.ceil(0.7 * len(bits)))
    train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
    train_parity, test_parity = parity[:traintest_cutoff], parity[traintest_cutoff:]

    washout=int(len(train_bits)/10)
    inputESN.train(train_bits, train_parity, washout=washout)

    ### generate new Training data for outputESN
    output_N = 1500
    bits, _, target = Parity_Data_Generator.generateParityData(output_N, n, timescale=timescale, randomstate=rng)
    traintest_cutoff = int(np.ceil(0.7 * len(bits)))
    train_bits, test_bits = bits[:traintest_cutoff], bits[traintest_cutoff:]
    train_targets, test_targets = target[:traintest_cutoff], target[traintest_cutoff:]

    train_predicted_parity = inputESN.predict(train_bits)
    test_predicted_parity = inputESN.predict(test_bits)
    # z-transformation
    output_scale = 1/np.std(train_predicted_parity)
    output_shift = -np.mean(train_predicted_parity)/np.std(train_predicted_parity)

    outputESN = ESN(#inputs=inputESN_reservoirSize, # for randomprojectionmatrix
                    inputs=1, # for squashing
                    neurons=outputESN_reservoirSize,
                    spectral_radius=params["output_spectral_rad"],
                    leak_rate= params["output_leak_rate"],  # adjust leak_rate
                    dt= 0.1,
                    sparsity=0.7,
                    noise=0.01,
                    input_scale=output_scale,
                    input_shift=output_shift)

    washout=int(len(test_predicted_parity/10))
    outputESN.train(train_predicted_parity,train_targets,washout=washout)
    final_prediction = outputESN.predict(test_predicted_parity)

#### randomprojectionmatrix
'''
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
'''
    errors[trial] = np.sqrt(np.mean((final_prediction - test_targets) ** 2))

averaged_error = errors.mean()
#save in a dictionary
all_outputs = dict(params=params,errors=errors,averaged_error=averaged_error)
# save params and errors and average
with open("/home/student/k/ktrebing/Documents/BA-ESN/scripterrors/esnError{:>05}.p".format(jobID), 'wb') as outputFile:
    pickle.dump(all_outputs, outputFile)

print("--- %s minutes ---" % (time.time() - start_time)/60)
