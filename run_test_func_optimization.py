import GPy
from model.stopping_criteria import *
from model.BayesianOptimization import TestFunctionOptimization
from utils import utils, bo_utils
from get_dataset import test_func
import random
import os


def define_stopping_criteria(threshold, rate, budget, delta, start_timing):
    pic = PICriterion(budget, start_timing=start_timing, threshold=threshold)
    eic_med = EICriterion(budget, start_timing=start_timing, rate=rate)
    rc_med = RegretCriterion(budget, delta, start_timing=start_timing, rate=rate)
    pc_med = ProposedCriterion(budget, delta, start_timing, threshold_type="med", rate=rate)
    pc_auto = ProposedCriterion(budget, delta, start_timing, threshold_type="auto")
    return [pic, eic_med, rc_med, pc_med, pc_auto]


def main():
    # experimental parameter
    n_iterate = 10
    start_sample_size = 20  # intial sample size
    budgets = [500, 1000, 500, 1000, 500, 500]  # budget of BO
    threshold = 0.01  # threshold of the stopping criterion based on PI
    rate = 0.01  # rate when determining the threshold based on median
    delta = 0.1  # confidence parameter of LCB
    max_iters = 10  # the number of iterations when the hyperparameter of GP is optimized
    acq_func_name = 'LCB'  # acuisition function ("LCB", "EI", "MPI")
    function_names = ["holder_table", "cross_in_tray", "six_hump_camel", "easom", "rosenbrock", "booth"]  # test function's name ("holder_table", "cross_in_tray", "matyas", "six_hump_camel", "bukin", "drop_wave")
    exact_feval = True  # True : noise free, False : noize added
    noise_var = 1e-6  # noise variance
    start_timing = 10  # sample size beginning a stop decision of BO

    for i, function_name in enumerate(function_names):
        seed = 2
        budget = budgets[i]
        dir = "result/BO_test_func/" + function_name + "/"
        os.makedirs(dir, exist_ok=True)
        np.random.seed(seed)
        random.seed(seed)
        for e in range(n_iterate):
            # set test function
            x_range, x_min = test_func.get_config(function_name)
            bounds, param_dim, param_range = bo_utils.set_bounds("test_func", x_range)
            f_min = test_func.function(x_min, x_min, function_name)[0]

            # define stopping criteria
            stopping_criteria = define_stopping_criteria(threshold, rate, budget, delta, start_timing)

            # set initial kernel
            kernel = GPy.kern.Matern52(lengthscale=1, variance=1, input_dim=len(bounds), ARD=False)

            # set parameters of BO
            tfo = TestFunctionOptimization(start_sample_size=start_sample_size, bounds=bounds, param_dim=param_dim,
                                           acq_func_name=acq_func_name, kernel=kernel, stopping_criteria=stopping_criteria,
                                           function_name=function_name, x_min=x_min, exact_feval=exact_feval,
                                           fixed_noise_var=noise_var, f_min=f_min, delta=delta)
            # execute BO
            tfo.explore(budget, max_iters)
            utils.serialize(tfo, dir+"tfo_{}_{}.dat".format(acq_func_name,e))


if __name__ == "__main__":
    main()
