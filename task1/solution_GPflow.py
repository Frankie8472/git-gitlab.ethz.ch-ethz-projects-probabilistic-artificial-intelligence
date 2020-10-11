import numpy as np
import gpflow
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

## for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

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
        """
            TODO: enter your code here
        """
        #self.k = gpflow.kernels.Matern52(variance=10.0, lengthscales=[0.5,0.5])
        self.m = None
        self.M = 50 # Number of inducing locations
        self.kernel = gpflow.kernels.SquaredExponential(variance=0.3, lengthscales=[30,1])
        self.minibatch_size = 200
        self.X = None
        self.Y = None
        self.Z = None
        pass

    def run_adam(self, model, iterations, train_dataset):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(self.minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        return logf       

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        pY, pYv = self.m.predict_y(test_x)
        pY = pY.numpy().reshape(100,)
        pYv = pYv.numpy().reshape(100,)


        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return pY

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        N = train_x.shape[0]
        self.X = train_x[:N,:]
        self.Y = train_y[:N].reshape((N,1))
        data = (self.X, self.Y)
        self.Z = self.X[:self.M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset
        self.m = gpflow.models.SVGP(self.kernel, gpflow.likelihoods.Gaussian(), self.Z, num_data=N)

        elbo = tf.function(self.m.elbo)
        tensor_data = tuple(map(tf.convert_to_tensor, data))
        elbo(tensor_data)  # run it once to trace & compile

        train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y)).repeat().shuffle(N)
        train_iter = iter(train_dataset.batch(self.minibatch_size))
        ground_truth = elbo(tensor_data).numpy()
        elbo(next(train_iter))
        evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, 100)]
    
        # We turn off training for inducing point locations
        gpflow.set_trainable(self.m.inducing_variable, False)
        maxiter = ci_niter(20000)

        logf = self.run_adam(self.m, maxiter, train_dataset)

        print_summary(self.m)
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
