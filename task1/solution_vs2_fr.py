import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler, Nystroem

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
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
        print("==> Initializing")
        kernel1 = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        kernel2 = DotProduct(sigma_0=0.0)
        self.nystroem = Nystroem(kernel=kernel1, random_state=1, n_components=20)
        self.model = GaussianProcessRegressor(kernel=kernel2, alpha=0.1**2, n_restarts_optimizer=0)
        return

    def predict(self, test_x):
        print("==> Predicting")
        print("====> Nystroem")
        test_x = self.nystroem.transform(test_x)
        print("====> MatMul")
        #test_x = test_x @ test_x.T
        print("====> GPR")
        return self.model.predict(test_x)

    def fit_model(self, train_x, train_y):
        print("==> Training")
        print("====> Nystroem")
        train_x = self.nystroem.fit_transform(train_x, train_y)
        print("====> MatMul")
        #train_x = train_x @ train_x.T
        print("====> GPR")
        self.model.fit(train_x, train_y)
        return


def main():
    test_size = None
    print("==> Loading")
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')
    M = Model()
    M.fit_model(train_x[:test_size], train_y[:test_size])
    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
