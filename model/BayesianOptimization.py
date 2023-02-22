import numpy as np
from utils import bo_utils
from tqdm import tqdm
from get_dataset import test_func
from model.GPyOptBO import GPyOptBO
from GPyOpt.core.task.space import Design_space
import statistics


class BaseBayesianOptimization():
    def __init__(self, start_sample_size, bounds, param_dim, acq_func_name, kernel, stopping_criteria, pool_size=1000, exact_feval=False, f_min=None, fixed_noise_var=1e-9, noise_range=None, delta=0.1):
        self.start_sample_size = start_sample_size
        self.bounds = bounds
        self.param_dim = param_dim
        self.acq_func_name = acq_func_name
        self.kernel = kernel.copy()
        self.kernel_topn = kernel.copy()
        self.stopping_criteria = stopping_criteria
        self.exact_feval = exact_feval
        self.f_min = f_min
        self.pool_size = pool_size
        self.fixed_noise_var = fixed_noise_var
        self.delta = delta
        if noise_range is None:
            self.noise_range = [1e-20, 1e5]
        else:
            self.noise_range = noise_range
        self.regret = np.empty(0, float)

    def explore(self, budget, max_iters):
        pass

    def set_bo(self, max_iters=0, fixed_noise_var=None, sqrt_beta=None):
        if fixed_noise_var is not None:
            params = bo_utils.get_bo_params(self.X, self.y, self.acq_func_name, self.bounds, kernel=self.kernel.copy(),
                                            max_iters=max_iters, exact_feval=True, sqrt_beta=sqrt_beta,
                                            fixed_noise_var=fixed_noise_var, noise_range=self.noise_range)
        else:
            params = bo_utils.get_bo_params(self.X, self.y, self.acq_func_name, self.bounds, kernel=self.kernel.copy(),
                                            max_iters=max_iters, exact_feval=self.exact_feval, sqrt_beta=sqrt_beta,
                                            fixed_noise_var=self.fixed_noise_var, noise_range=self.noise_range)
        bo = GPyOptBO(**params)
        return bo

    def set_stopping_criteria_param(self):
        pool_size = 1000
        constraints = None
        space = Design_space(self.bounds, constraints)
        candidate_param = bo_utils.sample_inputs(space.config_space, pool_size)
        for sc in self.stopping_criteria:
            sc.set_param(self.param_dim, candidate_param)

    def get_initial_sample(self,):
        pass

    def get_new_output(self, x_new):
        pass

    def update_dataset(self, x_new):
        y_new = self.get_new_output(x_new)
        self.X = np.concatenate([self.X, x_new], axis=0)
        self.y = np.concatenate([self.y, np.array([[y_new]])], axis=0)
        self.x_opt_indecies.append(np.argmin(self.y))

    def update_regret(self):
        f_t = self.y[self.x_opt_indecies[-1]]
        if self.f_min is not None:
            self.regret = np.append(self.regret, f_t - np.array([self.f_min]), axis=0)
        else:
            self.regret = np.append(self.regret, f_t, axis=0)


class HyperParameterOptimizationDiscrete(BaseBayesianOptimization):
    def __init__(self, *args, candidates, rmses, stds, test_scores, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = candidates
        self.rmses = rmses
        self.candididate_stds = stds
        self.candididate_test_scores = test_scores
        self.test_regret = np.empty(0, float)
        self.noise_var = np.empty(0, float)

    def explore(self,budget,max_iters):
        # sample initial dataset
        self.X, self.y, self.x_opt_indecies, self.stds, self.test_scores = self.get_initial_sample()
        self.set_stopping_criteria_param()
        for i in tqdm(range(budget)):
            # set configuration of current BO
            sqrt_beta = np.sqrt(2 * np.log(self.param_dim * self.X.shape[0] ** 2 * np.pi ** 2 / (6 * self.delta)))
            bo_new = self.set_bo(max_iters=max_iters, sqrt_beta=sqrt_beta)
            # explore new hyperparameter
            x_new = bo_new.suggest_next_locations()
            self.kernel = bo_new.model.model.kern.copy()

            self.update_regret()
            self.update_test_regret()
            # check stopping condition
            if i != 0:
                for sc in self.stopping_criteria:
                    if sc.name == "SR":
                        if sc.threshold_type == "cv":
                            threshold = self.stds[self.x_opt_indecies[-1]]
                        else:
                            if sc.start_timing < i:
                                threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                            else:
                                threshold = 0
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i, threshold, test_regret=self.test_regret[-1])
                    elif sc.name == "Ours":
                        if sc.threshold_type == "auto":
                            noise_var = bo_new.model.model.Gaussian_noise.variance[0]
                            self.noise_var = np.append(self.noise_var, noise_var)
                            sc.check_threshold(bo_old, bo_new, self.X, self.y, self.x_opt_indecies, self.regret[-1], i,
                                               threshold=None, test_regret=self.test_regret[-1], noise_var=noise_var)
                        else:
                            if sc.start_timing < i:
                                threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                            else:
                                threshold = 0
                            sc.check_threshold(bo_old, bo_new, self.X, self.y, self.x_opt_indecies, self.regret[-1], i,
                                               threshold=threshold, test_regret=self.test_regret[-1])
                    elif sc.name == "PI":
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i, test_regret=self.test_regret[-1])
                    elif sc.name == "EI":
                        if sc.start_timing < i:
                            threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                        else:
                            threshold = 0
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i, threshold, test_regret=self.test_regret[-1])

            bo_old = self.set_bo()
            bo_old._update_model()

            # update explored dataset
            self.update_dataset(x_new)

        for sc in self.stopping_criteria:
            sc.set_budget_timing(self.X, self.y, self.regret[-1], budget, test_regret=self.test_regret[-1])

    def get_new_output(self, x_new):
        dist = np.sum((self.candidates - x_new)**2,axis=1)
        index = np.argmin(dist)
        rmse = self.rmses[index]
        std = self.candididate_stds[index]
        test_score = self.candididate_test_scores[index]
        return rmse, std, test_score

    def get_initial_sample(self,):
        X = bo_utils.sample_inputs(self.bounds, self.start_sample_size)
        y = np.zeros(self.start_sample_size)
        stds = np.zeros(self.start_sample_size)
        test_scores = np.zeros(self.start_sample_size)
        for t,param in enumerate(X):
            y[t], stds[t], test_scores[t] = self.get_new_output(param[None, :])
        y = y[:, None]
        x_opt_indecies = [np.argmin(y)]
        return X, y, x_opt_indecies, stds, test_scores

    def update_dataset(self, x_new):
        y_new, std, test_score = self.get_new_output(x_new)
        self.X = np.concatenate([self.X, x_new], axis=0)
        self.y = np.concatenate([self.y, np.array([[y_new]])], axis=0)
        self.x_opt_indecies.append(np.argmin(self.y))
        self.stds = np.concatenate([self.stds, np.array([std])])
        self.test_scores = np.concatenate([self.test_scores, np.array([test_score])])


    def update_test_regret(self):
        f_t = np.array([self.test_scores[self.x_opt_indecies[-1]]])
        self.test_regret = np.append(self.test_regret, f_t, axis=0)


class TestFunctionOptimization(BaseBayesianOptimization):
    def __init__(self, *args, function_name, x_min, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_name = function_name
        self.x_min = x_min
        self.noise_var = np.empty(0, float)
        self.history = {}

    def explore(self,budget,max_iters):
        # sample initial dataset
        self.X, self.y, self.x_opt_indecies = self.get_initial_sample()
        self.set_stopping_criteria_param()
        self.history["length"] = np.zeros(budget)
        self.history["var"] = np.zeros(budget)
        self.history["noise_var"] = np.zeros(budget)
        for i in tqdm(range(budget)):
            # set configuration of current BO
            sqrt_beta = np.sqrt(2 * np.log(self.param_dim * self.X.shape[0] ** 2 * np.pi ** 2 / (6 * self.delta)))
            bo_new = self.set_bo(max_iters=max_iters, sqrt_beta=sqrt_beta)
            # explore new hyperparameter
            x_new = bo_new.suggest_next_locations(ignored_X=self.X)
            self.kernel = bo_new.model.model.kern.copy()
            self.history["length"][i] = self.kernel.lengthscale[0]
            self.history["var"][i] = self.kernel.variance[0]
            self.history["noise_var"][i] = bo_new.model.model.Gaussian_noise.variance[0]

            self.update_regret()
            # check stopping condition
            if i != 0:
                for sc in self.stopping_criteria:
                    if sc.name == "SR":
                        if sc.start_timing < i:
                            threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                        else:
                            threshold = 0
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i, threshold)
                    elif sc.name == "Ours":
                        if sc.threshold_type == "auto":
                            noise_var = bo_new.model.model.Gaussian_noise.variance[0]
                            self.noise_var = np.append(self.noise_var, noise_var)
                            sc.check_threshold(bo_old, bo_new, self.X, self.y, self.x_opt_indecies, self.regret[-1], i, threshold=None, noise_var=noise_var, variance=self.kernel.variance)
                        else:
                            if sc.start_timing < i:
                                threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                            else:
                                threshold = 0
                            sc.check_threshold(bo_old, bo_new, self.X, self.y, self.x_opt_indecies, self.regret[-1], i, threshold)
                    elif sc.name == "PI":
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i)
                    elif sc.name == "EI":
                        if sc.start_timing < i:
                            threshold = sc.rate * statistics.median(sc.seq_values[:sc.start_timing])
                        else:
                            threshold = 0
                        sc.check_threshold(bo_new, self.X, self.y, self.regret[-1], i, threshold)

            bo_old = self.set_bo(fixed_noise_var=bo_new.model.model.Gaussian_noise.variance[0])
            bo_old._update_model()

            # update explored dataset
            self.update_dataset(x_new)

        for sc in self.stopping_criteria:
            sc.set_budget_timing(self.X, self.y, self.regret[-1], budget)

    def get_new_output(self, x_new):
        noise_var = 0.0
        y_new = np.array(test_func.get_output(x_new,self.x_min,self.function_name, noise_var=noise_var))[:,None]
        return y_new

    def get_initial_sample(self):
        noise_var = 0.0
        X, y, x_min = test_func.get_dataset(self.start_sample_size, function_name=self.function_name, noise_var=noise_var)
        x_opt_indecies = [np.argmin(y)]
        return X, y, x_opt_indecies

    def update_dataset(self, x_new):
        y_new = self.get_new_output(x_new)
        self.X = np.concatenate([self.X, x_new], axis=0)
        self.y = np.concatenate([self.y, y_new], axis=0)
        self.x_opt_indecies.append(np.argmin(self.y))

