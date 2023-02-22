import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model.ridge_regression import RidgeRegression
from model.logistic_regression import NonlinearLogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import GPy


def sample_inputs(space, sample_size):
    sample = np.zeros((sample_size, len(space)))
    for dim, comp in enumerate(space):
        if comp["type"] == "continuous":
            sample[:, dim] = np.random.uniform(comp["domain"][0], comp["domain"][1], sample_size)
        elif comp["type"] == "discrete":
            indecies = np.random.randint(0, len(comp["domain"]), sample_size)
            sample[:, dim] = np.array(comp["domain"])[indecies]
        elif comp["type"] == "categorical":
            sample[:, dim] = np.random.randint(comp["domain"][0], comp["domain"][1], sample_size)
        else:
            sample[:, dim] = np.random.randint(comp["domain"][0], comp["domain"][1], sample_size)
    return sample


def set_model_param(params, model):
    if isinstance(model, KNeighborsClassifier):
        model.n_neighbbors = int(np.exp(params[0]))
        if int(params[1]) == 0:
            model.weight = "uniform"
        elif int(params[1]) == 1:
            model.weight = "distance"
        model.leaf_size = int(np.exp(params[2]))
        model.p = int(np.exp(params[3]))
    elif isinstance(model, GaussianProcessRegressor):
        if int(params[0]) == 0:
            nu = 0.5
        elif int(params[0]) == 1:
            nu = 1.5
        elif int(params[0]) == 2:
            nu = 2.5
        else:
            nu = float("inf")
        kernel = np.exp(params[2])*Matern(np.exp(params[1]),nu=nu)
        model.kernel_ = kernel
        model.alpha = np.exp(params[3])
    elif isinstance(model, RidgeRegression):
        model.basis_size = int(np.exp(params[0]))
        model.length = np.exp(params[1])
        model.alpha = np.exp(params[2])
    elif isinstance(model, NonlinearLogisticRegression):
        model.basis_size = int(np.exp(params[0]))
        model.length = np.exp(params[1])
        model.C = np.exp(params[2])
    elif isinstance(model, SVR):
        model.gamma = np.exp(params[0])
        model.C = np.exp(params[1])
        model.epsilon = np.exp(params[2])
    elif isinstance(model,SVC):
        model.kernel = "sigmoid"
        model.gamma = np.exp(params[0])
        model.coeff0 = np.exp(params[1])
        model.C = np.exp(params[2])
    elif isinstance(model, RandomForestRegressor):
        model.n_estimators = int(np.exp(params[0]))
        model.max_depth = int(np.exp(params[1]))
        model.min_samples_split = np.exp(params[2])
    elif isinstance(model, RandomForestClassifier):
        model.n_estimators = int(np.exp(params[0]))
        model.max_depth = int(np.exp(params[1]))
        model.min_samples_split = np.exp(params[2])
    return model


def set_model(model_name):
    if model_name == "ridge":
        model = RidgeRegression()
        metric = "rmse"
    elif model_name == "svr":
        model = SVR(kernel="rbf")
        metric = "rmse"
    elif model_name == "gpr":
        model = GaussianProcessRegressor(optimizer=None)
        metric = "rmse"
    elif model_name == "rfr":
        model = RandomForestRegressor()
        metric = "rmse"
    elif model_name == "logistic":
        model = NonlinearLogisticRegression()
        metric = "accuracy"
    elif model_name == "svc":
        model = SVC(kernel="rbf")
        metric = "accuracy"
    elif model_name == "rfc":
        model = RandomForestClassifier()
        metric = "accuracy"
    elif model_name == "knnc":
        model = KNeighborsClassifier()
        metric = "accuracy"
    else:
        model = KNeighborsClassifier()
        metric = "accuracy"
    bounds,param_dim,param_range = set_bounds(model_name)

    return model,bounds,param_dim,param_range,metric


def model_evaluation(params, model, trainset, testset, n_splits=10, metric="rmse"):
    model = set_model_param(params, model)
    scores = np.empty(0,float)
    ss = KFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in ss.split(trainset[0], trainset[1]):
        X_train, X_test = trainset[0][train_index], trainset[0][test_index]
        y_train, y_test = trainset[1][train_index], trainset[1][test_index]
        model.fit(X_train, y_train)
        output = model.predict(X_test)
        if metric == "rmse":
            scores = np.append(scores,np.sqrt(np.mean((output-y_test)**2)))
        elif metric == "accuracy":
            scores = np.append(scores, 1 - accuracy_score(output, y_test))
    rmse = scores.mean()
    s2_cv = scores.var()
    std = np.sqrt((1.0/n_splits+1.0/(n_splits-1))*s2_cv)
    output = model.predict(testset[0])
    if metric == "rmse":
        test_score = np.sqrt(np.mean((output-testset[1])**2))
    elif metric == "accuracy":
        test_score = 1 - accuracy_score(output, testset[1])
    return rmse, std, test_score


def set_bounds(param_name, param_range=None):
    if param_name == "svr":
        param_range = [np.linspace(np.log(1e-3), np.log(1e3), 20), np.linspace(np.log(1e-3), np.log(1e3), 20), np.linspace(np.log(1e-3), np.log(1e3), 20)]
        param_dim = len(param_range)
        bounds = [{'name': 'gamma', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'C', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'epsilon', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "svc":
        param_range = [np.linspace(np.log(1e-3), np.log(1e3), 20), np.linspace(np.log(1e-3), np.log(1e3), 20), np.linspace(np.log(1e-3), np.log(1e3), 20)]
        param_dim = len(param_range)
        bounds = [{'name': 'gamma', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'coef0', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'C', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "ridge":
        param_range = np.array([np.linspace(np.log(2), np.log(30), 29), np.linspace(np.log(0.01), np.log(100), 20), np.linspace(np.log(0.01), np.log(100), 20)])
        param_dim = len(param_range)
        bounds = [{'name': 'basis_size', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'length_scale', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'reguralization', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "logistic":
        param_range = np.array([np.linspace(np.log(2), np.log(30), 29), np.linspace(np.log(0.01), np.log(100), 20), np.linspace(np.log(0.01), np.log(100), 20)])
        param_dim = len(param_range)
        bounds = [{'name': 'basis_size', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'length_scale', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'reguralization', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "rfr":
        param_range = [np.linspace(np.log(1), np.log(20), 20), np.linspace(np.log(1), np.log(20), 20), np.linspace(np.log(0.01), np.log(0.5), 20)]
        param_dim = len(param_range)
        bounds = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'max_depth', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'min_samples_split', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "rfc":
        param_range = [np.linspace(np.log(1), np.log(20), 20), np.linspace(np.log(1), np.log(20), 20), np.linspace(np.log(0.01), np.log(0.5), 20)]
        param_dim = len(param_range)
        bounds = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (param_range[0])},
                  {'name': 'max_depth', 'type': 'discrete', 'domain': (param_range[1])},
                  {'name': 'min_samples_split', 'type': 'discrete', 'domain': (param_range[2])}]
    elif param_name == "test_func":
        param_dim = len(param_range)
        bounds = [{'name': "x_0", 'type': 'continuous', 'domain': (param_range[0])},
                  {'name': "x_1", 'type': 'continuous', 'domain': (param_range[1])}]
    else:
        bounds = []
        param_range = []
        param_dim = 0
    return bounds,param_dim,param_range


def get_bo_params(X,y,acq_func_name,bounds,fixed_noise_var=1e-9,kernel=None,max_iters=0,exact_feval=False,noise_range=None,sqrt_beta=None):
    if kernel is None:
        kernel = GPy.kern.Matern52(input_dim=len(bounds), ARD=False)
    params = {'acquisition_type': acq_func_name,
              'acquisition_weight': sqrt_beta,
              'kernel': kernel,
              'domain': bounds,
              "f": None,
              'model_type': "modifiedGP",
              'X': X,
              'Y': y,
              'max_iters': max_iters,
              'optimize_restarts': 1,
              'fixed_noise_var': fixed_noise_var,
              'exact_feval': exact_feval,
              'de_duplication': True,
              "normalize_Y": False,
              "noise_range": noise_range,
              }
    if max_iters == 0:
        params['max_iters'] = 0
    return params

