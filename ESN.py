import numpy as np
from BlockedLeastSquares import BlockedLeastSquares


class ESN(object):
    def __init__(self, neurons, inputs, leak_rate, spectral_radius, dt, sparsity, noise, input_scale=1, input_shift=0):
        self.W_rec = np.random.randn(neurons, neurons)                  # recurrent matrix
        self.W_rec[np.random.rand(*self.W_rec.shape) < sparsity] = 0    # force sparsity
        self.W_rec /= np.abs(np.linalg.eigvals(self.W_rec)).max()
        self.W_rec *= spectral_radius                                   # enforce spectral radius
        self.W_in = np.random.randn(neurons, inputs) / np.sqrt(inputs)  # inputmatrix scaled by number of inputs
        # annahme: input-werte haben standard deviation = 1
        self.state = np.random.rand(neurons, 1) * 2 - 1
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.dt = dt
        self.sparsity = sparsity
        self.noise = noise
        self.neurons = neurons
        self.input_scale = input_scale
        self.input_shift = input_shift

    def _state_update(self, inp):
        # self.state += self.dt * (-self.leak_rate * self.state + np.tanh(
        #     self.W_in.dot(inp.reshape((-1, 1))*self.input_scale+self.input_shift) + self.W_rec.dot(self.state))) + self.noise * np.random.randn(
        #     self.neurons, 1)
        # performance boost:
        self.state += self.dt * (-self.leak_rate * self.state + np.tanh(
            inp.reshape((-1,1)) + self.W_rec.dot(self.state))) + self.noise * np.random.randn(self.neurons, 1)
        self.state[0] = 1   # bias term

    def train(self, data, targets, washout=0, block_size=10000):
        lstsq = BlockedLeastSquares()
        block_size = min(block_size, data.shape[0] - washout)
        block_num = int(np.ceil((data.shape[0] - washout) / block_size))
        # print(block_num)
        # wash out initial network state
        washoutblock = self.W_in.dot(data[:washout, :].T*self.input_scale+self.input_shift)
        for i, row in enumerate(washoutblock.T):
            self._state_update(row)

        # for every block calculate intermediate result
        for j in range(block_num):
            block = data[washout + j * block_size:washout + (j + 1) * block_size, :]
            state_values = np.empty((block.shape[0], self.neurons))
            block = self.W_in.dot(block.T*self.input_scale+self.input_shift)
            for i, row in enumerate(block.T):
                self._state_update(row)
                state_values[i, :] = self.state.ravel()
            lstsq.update(targets[washout + j * block_size:washout + (j + 1) * block_size, :], state_values)

        # self.readout_weights = np.linalg.lstsq(state_values, targets[washout:, :])[0].T
        self.readout_weights = lstsq.omega.T # calculate final result and set weights

        # predictions = self.predict(data)
        # error = np.sqrt(((predictions - targets) ** 2 / targets.var(axis=0)).mean())
        # return error

    def predict(self, data, readout_weights=None):
        if readout_weights == None:
            readout_weights = self.readout_weights

        block = self.W_in.dot(data.T * self.input_scale + self.input_shift)
        readout_values = np.empty((data.shape[0], readout_weights.shape[0]))
        for i, row in enumerate(block.T):
            self._state_update(row)
            readout_values[i, :] = readout_weights.dot(self.state).ravel()
        return readout_values