import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import fmin_l_bfgs_b

domain = np.array([[0, 5]])

# TODO machen das man es reproduzieren kann
""" Solution """


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.seed = 42

        self.domain = np.array([[0.0, 5.0]])

        self.f_sigma = 0.5
        self.f_lengthscale = 0.5
        self.f_nu = 2.5

        self.v_sigma = np.sqrt(2)
        self.v_lengthscale = 0.5
        self.v_nu = 2.5
        self.v_mean = 1.5
        self.v_min = 1.2

        self.x = np.random.rand(4, 1) * 5
        self.y = None
        self.v = None

        self.f_kernel = self.f_sigma ** 2 * Matern(length_scale=self.f_lengthscale, length_scale_bounds='fixed', nu=self.f_nu)
        self.f_model = GaussianProcessRegressor(kernel=self.f_kernel, random_state=self.seed)

        self.v_kernel = self.v_mean + (self.v_sigma ** 2 * Matern(length_scale=self.v_lengthscale, length_scale_bounds='fixed', nu=self.v_nu))
        self.v_model = GaussianProcessRegressor(kernel=self.v_kernel, random_state=self.seed)
        return

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if self.y is None:
            return self.x

        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])  # [[somenumber]]

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        y_mean, y_std = self.f_model.predict(x.reshape(1, -1), return_std=True)
        y_sample_mean = self.f_model.predict(self.x)

        y_sample_mean_opt = np.max(y_sample_mean)

        beta = 0.0001
        af_value = y_mean[0] + beta * y_std[0]
        return af_value

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.x = np.vstack((self.x, x))

        if self.y is None and self.v is None:
            self.y = f
            self.v = v
        else:
            self.y = np.vstack((self.y, f))
            self.v = np.vstack((self.v, v))

        self.f_model.fit(self.x, self.y)
        self.v_model.fit(self.x, self.v)
        return

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        # TODO: enter your code here
        y_sol = 0
        v_sol = 0
        for idx, item in enumerate(self.x):
            if item < y_sol and self.v[idx] >= self.v_min:
                y_sol = item
                v_sol = self.v[idx]

        return y_sol


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
