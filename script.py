import Parity_Data_Generator
import numpy as np
from newESN import ESN
import pickle
import os
import itertools

# get Parameter according to JobID
def generateParams(ID):
    all = itertools.product(np.linspace(0.1, 1.5, num=10),
                            np.linspace(0.1, 2.0, num=10),
                            np.linspace(0.1, 1.5, num=10),
                            np.linspace(0.1, 2.0, num=10))
    # helper recipe nth
    values = next(itertools.islice(all,ID, None))
    keys = "input_leak_rate", "input_spectral_rad", "output_leak_rate", "output_spectral_rad"
    return dict(zip(keys,values))

# get JobID
jobID = int(os.environ.get("SGE_TASK_ID"))-1

params = generateParams(jobID)

N = 4000   # number of datapoints
n = 3       # n-parity
timescale = 50
rng = np.random.RandomState(1578) # maybe other seed 
inputESN_reservoirSize=300
outputESN_reservoirSize= 50

averages = 2

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
    inputESN_ident_Matrix=np.eye(inputESN_reservoirSize)

    outputESN = ESN(inputs=inputESN_reservoirSize,
                    neurons=outputESN_reservoirSize,
                    spectral_radius=params["output_spectral_rad"],
                    leak_rate= params["output_leak_rate"],  # adjust leak_rate
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

    errors[trial] = np.sqrt(np.mean((final_prediction - test_targets) ** 2))

averaged_error = errors.mean()

all_outputs = dict(params=params,errors=errors,averaged_error=averaged_error)
# save params and errors and average
with open("/home/student/k/ktrebing/Documents/BA-ESN/scripterrors/esnError{:>05}.p".format(jobID), 'wb') as outputFile:
    pickle.dump(all_outputs, outputFile)
