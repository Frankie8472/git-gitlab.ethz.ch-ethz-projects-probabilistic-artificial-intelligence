import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

# Constants for the following functions
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04
SEED = 42
CORES = 4


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted >= true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted >= THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted <= true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    reward = W4 * np.logical_and(predicted < THRESHOLD, true < THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model:
    """Model for Task1 Gaussian process regression (GPR).

        This model tackles four challenges:
            1. Model selection: With a combination of a Linear-, RBF- and
               Whitenoise-Kernel the GP should be modelled faithfully.
            2. Large scale learning: To overcome the computationally intensivenenss,
               we used the Bagging method from sklearn (without bootstrapping).
            3. Asymmetric costs: To ensure that there are no false positives,
               we add the std to the output mean to only get false negatives.

        """

    def __init__(self):
        """Inits Model with a kernel, a classifier (gpr) and a bagging method."""
        print("==> Initializing")
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=SEED
        )
        self.model = BaggingRegressor(
            base_estimator=gpr,
            n_estimators=20,
            max_samples=1000,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            verbose=0,
            n_jobs=CORES,
            random_state=SEED
        )
        return

    def predict(self, test_x):
        """Performs a upper boundary prediction on a given test set (test_x)."""
        print("==> Predicting")
        y_preds = list()
        for model in self.model.estimators_:
            y_pred_mean, y_pred_std = model.predict(test_x, return_std=True)
            y_pred = y_pred_mean + 1.195 * y_pred_std
            y_preds.append(y_pred)
        return np.asarray(y_preds).mean(axis=0)

    def fit_model(self, train_x, train_y):
        """Fits the model on the given train set (train_x) for the given output (train_y)."""
        print("==> Training")
        self.model.fit(train_x, train_y)
        return


def main():
    """Used for local testing only."""
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
