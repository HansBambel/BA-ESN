"""
A block-matrix method for solving a large regularized linear system of equations.

1/2/2016, Johannes Leugering
"""
import numpy as np
from numpy.linalg import inv

class BlockedLeastSquares(object):
    """Implements a block-matrix method for solving a large regularized linear system of equations.

    The system to be approximately solved for the optimal parameter vector w (omega) is given by:
        y = A w

    The normal equation solved instead to find the least squares approximation is given by:
        A' P y + Q w_0 = (A' P A + Q)
    =>  w = (A' P A + Q)^{-1} (A' P y + Q w_0)
    where:
        P holds the expected inverse covariance matrix of the data block A,
        Q is the regularizer determining the correlation and scale of the found parameter vector,
        w_0 is the expected value of the parameter vector,
        y holds the target values to be approximated and
        A holds the training data.

    A and y are assumed to be of the block matrix form:

        | A_0 |       | y_0 |
    A = |  :  |,  y = |  :  |
        | A_n |       | y_n |

    Model predictions are generated for a matrix of testing data A_p by the linear model as follows:
        y_p = A_p * w

    Example
    -------
    The simplest usage of the function is to leave the regularization parameters at their default values.

    >>> lstsq = BlockedLeastSquares()
    >>> for i in range(numer_of_blocks):
    ...    # generate training data block A_i and target block y_i
    ...    A_i = ...
    ...    y_i = ...
    ...    # update the model
    ...    lstsq.update(A_i, y_i)
    ... # Generate block of testing data A_t
    ... A_t = ...
    ... # Generate a prediction for y_t
    ... y_t = lstsq(A_t)
    ... # Print the parameters
    ... print(lstsq.omega)
    

    Attributes:
    -----------
    omega : ndarray
        the parameter vector fitted by the model (w)
    residuals : ndarray
        the residuals calculated on the training data
    error : float
        the error calculated on the training data
    """

    def __init__(self, Q=None, omega_0=None):
        """Initializes the model.
        
        Parameters
        ----------
        Q : Optional[ndarray]
            the inverse covariance matrix of the regularizer. Default `None` implies no regularization.
        omega_0 : Optional[ndarray]
            the expected value of the parameter vector. Default `None` results in the 0-vector.
        """

        self.Q = Q
        self.omega_0 = omega_0
        self._LHS = None
        self._RHS = None
        self.omega = None
        self.residuals = None
        self.error = None

    def update(self, target_block, data_block, inv_cov_block=None):
        """Updates the model with a new block of training targets and data.

        Parameters
        ----------
        target_block : ndarray
            `n*k` matrix containing targets for `n` samples of `k` predicted variables
        data_block : ndarray
            `n*m` matrix containing training data for `n` samples of `m` features (e.g. m-dimensional ESN state)
        inv_cov_block : Optional[ndarray]
            inverse of the expected covariance matrix of an `n*m` data block; 
            the default value None implies an identity covariance matrix (i.e. independent samples)

        Returns
        -------
        ndarray
            the model's resulting parameter vector `self.omega`
        """
        # initialize LHS and RHS if necessary
        if self._LHS == None:
            if self.Q == None:
                self._LHS = np.eye(data_block.shape[1])*1e-8
            elif np.isscalar(self.Q):
                self._LHS = np.eye(data_block.shape[1])*self.Q
            else:
                self._LHS = self.Q

            if self.omega_0 == None:
                self._RHS = np.zeros((data_block.shape[1],target_block.shape[1]))
            elif np.isscalar(self.omega_0):
                self._RHS = self._LHS.dot(np.zeros((data_block.shape[1],target_block.shape[1]))*self.omega_0)
            else:
                self._RHS = -self._LHS.dot(self.omega_0)

        # initialize inv_cov_block if necessary:
        if inv_cov_block == None:
            inv_cov_block = 1#np.eye(data_block.shape[0])

        # Update LHS and RHS
        BP = data_block.T.dot(inv_cov_block)
        self._LHS += BP.dot(data_block)
        self._RHS += BP.dot(target_block)

        self.omega = inv(self._LHS).dot(self._RHS)
        self.residuals = self(data_block) - target_block
        self.error = self.rmse(self.residuals)
        return self.omega

    def __call__(self, data_block):
        """Calling the model with a data block returns the resulting model prediction.

        Parameters
        ----------
        data_block : ndarray
            `i*m` matrix containing data on which to base the model prediction.
            `m` must correspond to the number of fitted parameters

        Returns
        -------
        ndarray
            `i*k` matrix of the model predictions.
            `k` is the number of predicted variables
        """
        return data_block.dot(self.omega)

    def reset(self):
        """Reset the model to its initial state."""
        self._RHS = None
        self._LHS = None

    @staticmethod
    def rmse(v):
        """Convenience class method to calculate the root mean squared error (RMSE)"""
        return np.linalg.norm(v)/np.sqrt(len(v))
