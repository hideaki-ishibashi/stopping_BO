from model.stopping_criteria import *
from model.BayesianOptimization import HyperParameterOptimizationDiscrete
from utils import utils, bo_utils
import random
import GPy
import os


def define_stopping_criteria(threshold, rate, budget, delta, start_timing):
    rc_med = RegretCriterion(budget, delta, start_timing=start_timing, rate=rate)
    rc_cv = RegretCriterion(budget, delta, start_timing=start_timing, rate=rate, threshold_type="cv")
    eic_med = EICriterion(budget, start_timing=start_timing, rate=rate)
    pic = PICriterion(budget, start_timing=start_timing, threshold=threshold)
    pc_med = ProposedCriterion(budget, start_timing=start_timing, threshold_type="med", rate=rate)
    pc_auto_med = ProposedCriterion(budget, delta, start_timing=start_timing, threshold_type="auto")
    return [pic, eic_med, rc_med, rc_cv, pc_med, pc_auto_med]


def main():
    # experimental parameter
    n_iterate = 10  # sample size beginning a stop decision of BO
    start_sample_size = 10  # initial sample size
    threshold = 0.01  # threshold of the stopping criterion based on PI
    rate = 0.01  # rate when determining the threshold based on median
    delta = 0.1  # confidence parameter of LCB
    budget = 500  # budget of BO
    max_iters = 10  # the number of iterations when the hyperparameter of GP is optimized
    acq_func_name = "LCB"  # acuisition function ("LCB", "EI")
    model_names = ["ridge", "svr", "rfr", "logistic","svc","rfc"]
    data_names = {"ridge": ["gas_turbine", "power_plant", "protein"], "svr": ["gas_turbine", "power_plant", "protein"],
                  "rfr": ["gas_turbine", "power_plant", "protein"], "logistic": ["skin", "HTRU2", "electrical_grid_stability"],
                  "svc": ["skin", "HTRU2", "electrical_grid_stability"], "rfc": ["skin", "HTRU2", "electrical_grid_stability"]}
    exact_feval = True  # True : noise free, False : noize added
    noise_var = 1e-6  # noise variance
    start_timing = 10  # sample size beginning a stop decision of BO
    seed = 2

    for model_name in model_names:
        for data_name in data_names[model_name]:
            np.random.seed(seed)
            random.seed(seed)
            dir = "result/HPO_experiments/" + data_name + "/" + model_name+"/"
            os.makedirs(dir, exist_ok=True)
            for e in range(n_iterate):
                # get model, space of hyper-parameter and evaluation metric of the predicive model
                model, bounds, param_dim, param_range, metric = bo_utils.set_model(model_name)
                candidates = np.loadtxt(dir + "discretized_param.txt")
                rmses = np.loadtxt(dir + "discretized_rmse.txt")
                stds = np.loadtxt(dir + "discretized_std.txt")
                test_scores = np.loadtxt(dir + "discretized_test_score.txt")
                f_min = np.min(rmses)

                # define stopping criteria
                stopping_criteria = define_stopping_criteria(threshold, rate, budget, delta, start_timing)

                # set initial kernel
                kernel = GPy.kern.Matern52(lengthscale=1, variance=1, input_dim=len(bounds), ARD=True)
                # set parameters of BO
                hpo = HyperParameterOptimizationDiscrete(start_sample_size=start_sample_size, bounds=bounds, param_dim=param_dim,
                                                         acq_func_name=acq_func_name, kernel=kernel, stopping_criteria=stopping_criteria,
                                                         f_min=f_min, candidates=candidates, rmses=rmses, stds=stds,
                                                         exact_feval=exact_feval, test_scores=test_scores, fixed_noise_var=noise_var, delta=delta)
                # execute BO
                hpo.explore(budget, max_iters=max_iters)
                utils.serialize(hpo,dir+"hpo_{}_{}.dat".format(acq_func_name, e))


if __name__ == "__main__":
    main()
