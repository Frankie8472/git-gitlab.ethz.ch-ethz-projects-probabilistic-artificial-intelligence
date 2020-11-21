import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm



domain = np.array([[0, 5]])

# TODO machen das man es reproduzieren kann
""" Solution """
 

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        #self.sigma_f_noise = 0.15
        
        self.f_sigma = 0.5
        self.f_lengthscale = 0.5
        self.f_smoothness = 2.5

        #self.sigma_v_noise = 0.0001

        self.v_sigma = np.sqrt(2)
        self.v_lengthscale = 0.5
        self.v_smoothness = 2.5
        self.v_mean = 1.5 # TODO verstehen wo dieser gebraucht wird 
        
        #self.X = np.random.randint(domain[0][0],domain[0][1] , size=2).reshape(-1,1)
        self.X = np.random.rand(4,1)*5 #np.array([[1],[3]])
        self.y = np.asarray([[f(x)] for x in self.X]) # np.random.normal(0,self.sigma_f_noise) 
        self.v = np.asarray([[v(x)] for x in self.X]) # np.random.normal(0,self.sigma_v_noise)

        kernel = self.f_sigma**2 * Matern(length_scale=self.f_lengthscale, nu=self.f_smoothness) # TODO checken ob sigma squared und ob richtig eingesetzt
        self.model =  GaussianProcessRegressor(kernel)   
        self.model.fit(self.X,self.y) 

        kernel_v = ( self.v_sigma**2 * Matern(length_scale=self.v_lengthscale, nu=self.v_smoothness) ) + self.v_mean # TODO checken ob sigma squared und ob richtig eingesetzt und auch v_mean
        self.model_v =  GaussianProcessRegressor(kernel_v)   
        self.model_v.fit(self.X,self.v)

        pass


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
        actual  = self.optimize_acquisition_function() # TODO Hier müssen wir irgendwie die Bound von 1.2 durchsetzten 
        # aber trotzdem zum optimum gelangen, im text steht man kann zwischen durch unter die bound von 1.2 gehen...
        """
        if(v(actual[0]) < 1.2):
            if v(actual[0] + 0.5) > v(actual[0]):
                 actual[0] += 0.5
            else: 
                actual[0] -= 0.5
        """
        return actual
        #raise NotImplementedError


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
        return np.atleast_2d(x_values[ind]) # [[somenumber]]

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
        y , std = self.model.predict(x.reshape(1,-1), return_std = True)  
        speed, _ = self.model_v.predict(x.reshape(1,-1), return_std = True) # hier müssen wir den log nehemen ?...
        """
        speed_ = speed[0][0]
        p = 1
        if(speed_ < 1.2):
            p = (speed_ / 1.2)**2
        """
        af_value = (-y[0] + 0.0001*std)  #* p 
        #print(af_value)
        """"
        y_sample = self.model.predict(self.X)

        std = std.reshape(-1, 1)
        
        y_sample_opt = np.max(y_sample)
        imp = y - y_sample_opt - 0.01
        Z = imp / std
        af_value = imp * norm.cdf(Z) + std * norm.pdf(Z)
        """

        return af_value[0]
        #raise NotImplementedError



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
        self.X = np.vstack((self.X,x))
        self.y = np.vstack((self.y,f)) # @295 no we don't have to add the noise +np.random.normal(0,self.sigma_f_noise)
        self.v = np.vstack((self.v,v)) #+np.random.normal(0,self.sigma_v_noise)

        self.model.fit(self.X,self.y)
        self.model_v.fit(self.X,self.v)
        #raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        # TODO: enter your code here
        solution = 0
        v = 0
        for idx,item in enumerate (self.X):
            #print("item:       ", item)
            #print("f(item):    ",f(item))
            #print("v:          ", self.v[idx])
            #print("                ")
            if f(item) < solution and self.v[idx] >= 1.2: # Hier muss die bound von 1.2 zwingend halten 
                solution = item
                v = self.v[idx]
        #print("choosen:   ",solution)
        #print("f(item):    ",f(solution))
        #print("v:          ", v)
        return solution 
        #raise NotImplementedError


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