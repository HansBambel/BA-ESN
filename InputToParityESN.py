import numpy as np
from pyESN import ESN
import Parity_Data_Generator

class slowESN():
    def __init__(self, arity=3, randomState=np.random.RandomState(42)):
        '''
            good parameters for n=3:
                reservoir = 500
                spectral_radius = >0.75 (from 0.9 test error is 0.0)

            n=4:
                 reservoir = 1000
                  spectral_radius = 0.95
        '''
        if arity == 3:
            self.slowESN = ESN(n_inputs=1,
                          n_outputs=1,
                          n_reservoir=500,  # from 200 onwards test error decreases significantly
                          spectral_radius=0.85,  # from 0.9 the test error is 0.0
                          sparsity=0.95,
                          noise=0.001,
                          input_shift=0,
                          input_scaling=3,  # 3 seems to be fine. Maybe use lower
                          teacher_scaling=1.12,
                          teacher_shift=-0.7,
                          out_activation=np.tanh,
                          inverse_out_activation=np.arctanh,
                          random_state=randomState,
                          silent=False)
        elif arity == 4:
            self.slowESN = ESN(n_inputs=1,
                          n_outputs=1,
                          n_reservoir=1000,  # from 200 onwards test error decreases significantly
                          spectral_radius=0.95,  # from 0.9 the test error is 0.0
                          sparsity=0.95,
                          noise=0.001,
                          input_shift=0,
                          input_scaling=3,  # 3 seems to be fine. Maybe use lower
                          teacher_scaling=1.12,
                          teacher_shift=-0.7,
                          out_activation=np.tanh,
                          inverse_out_activation=np.arctanh,
                          random_state=randomState,
                          silent=False)

    def fit(self, train, target):
        pred_train = self.slowESN.fit(train, target)

    def predict(self, inputs):

    # print("Test error:")
        pred_test = self.slowESN.predict(inputs)
        better_pred_test = []
        for x in pred_test:
            better_pred_test.append(0 if x < 0.5 else 1)
        return np.array(better_pred_test)
        # print(better_pred_test)


######## For testing #########
