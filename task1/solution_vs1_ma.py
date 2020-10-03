import numpy as np
# import some already implemented GPR methodes (hope this is allowed)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, ExpSineSquared
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.01


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted > THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted < true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    # reward for correctly identified safe regions
    reward = W4 * np.logical_and(predicted <= THRESHOLD, true <= THRESHOLD)

    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        """
            TODO: enter your code here
        """
        self.noise = 0.1
        self.rbf =   ConstantKernel(1) + RBF(length_scale=1) * ConstantKernel(1) 
        self.gpr = None
        self.nystroem = None
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below 
        y = self.gpr.predict(test_x)
        
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        n_components = 2000
        self.nystroem = Nystroem(self.rbf,random_state=RANDOM_STATE,n_components=n_components) 
        self.nystroem.fit(train_x,train_y)
        idx = self.nystroem.component_indices_[:n_components]
        gpr = GaussianProcessRegressor(kernel=self.rbf, alpha=self.noise**2)
        gpr.fit(train_x[idx],train_y[idx])
        self.gpr = gpr 
        pass


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
