import numpy as np
from utils import utils
from GPyOpt.util.general import get_quantiles
from scipy.stats import norm


class BaseCriterion(object):
    def __init__(self, name, budget, start_timing=10):
        self.name = name
        self.config_name = ""
        self.stop_flags = False
        self.stop_timings = budget
        self.seq_values = np.empty(0, float)
        self.seq_thresholds = np.empty(0, float)
        self.start_timing = start_timing
        self.ei_diff = None
        self.kappa = None
        self.KL = None
        self.delta_f = None

    def reset(self, budget):
        self.stop_flags = False
        self.stop_timings = budget
        self.seq_values = np.empty(0, float)
        self.seq_thresholds = np.empty(0, float)


class ProposedCriterion(BaseCriterion):
    def __init__(self, budget, delta=0.1, start_timing=10, rate=0.1, threshold_type="auto"):
        super(ProposedCriterion, self).__init__("Ours",budget,start_timing)
        self.rate = rate
        self.delta = delta
        self.ei_diff = np.empty(0, float)
        self.kappa = np.empty(0, float)
        self.KL = np.empty(0, float)
        self.delta_f = np.empty(0, float)
        self.threshold_type = threshold_type
        self.threshold1 = np.empty(0, float)
        self.threshold2 = np.empty(0, float)

    def set_param(self, param_dim, candidate_param):
        self.param_dim = param_dim
        self.config_name = "-{}".format(self.threshold_type)
        self.candidate_param = candidate_param.copy()

    def calc_delta_f(self,f_opt_old,f_opt_new):
        delta_f = f_opt_old - f_opt_new
        return delta_f

    def calc_UCB(self,bo,X,sqrt_beta):
        mu_tr, var_tr = bo.model.model.predict_noiseless(X, full_cov=False)
        std_tr = np.sqrt(var_tr)
        return mu_tr+sqrt_beta*std_tr

    def calc_LCB(self,bo,X,sqrt_beta):
        candidate_param = np.concatenate([X, self.candidate_param], axis=0)
        mu_c, var_c = bo.model.model.predict_noiseless(candidate_param, full_cov=False)
        std_c = np.sqrt(var_c)
        return mu_c-sqrt_beta*std_c

    def calc_kappa(self, bo, X, sqrt_beta):
        UCB = self.calc_UCB(bo, X, sqrt_beta)
        LCB = self.calc_LCB(bo, X, sqrt_beta)
        return np.min(UCB) - np.min(LCB)

    def calc_KL(self, bo_old, bo_new, x_new, y_new):
        pos_old = bo_old.model.model.predict_noiseless(x_new, full_cov=True)
        noise_var = bo_new.model.model.likelihood.gaussian_variance()[0]
        KL = utils.calcKL_qp_fast(pos_old, y_new, 1 / noise_var)
        return KL

    def check_threshold(self, bo_old, bo_new, X, y, x_opt_indecies, regret, current_time, threshold, noise_var=None,
                        **kwargs):
        KL = self.calc_KL(bo_old, bo_new, X[-1][None, :], y[-1, 0])
        pos_old = bo_old.model.model.predict_noiseless(X[:-1], full_cov=False)
        pos_new = bo_new.model.model.predict_noiseless(X, full_cov=False)
        sqrt_beta = np.sqrt(2 * np.log(self.param_dim * X.shape[0] ** 2 * np.pi ** 2 / (6 * self.delta)))
        index = x_opt_indecies[-1]
        index_old = x_opt_indecies[-2]
        delta_f = np.abs(pos_old[0][index_old, 0] - pos_new[0][index, 0])
        kappa = self.calc_kappa(bo_old, X[:-1], sqrt_beta)
        if index == index_old:
            ei_diff = 0
        else:
            x_list = X[[index, index_old]]
            m, sigma = bo_new.model.model.predict_noiseless(x_list, full_cov=True)
            g = m[0] - m[1]
            if sigma[0,0]-2*sigma[0,1]+sigma[1,1] < 0:
                beta = 0
                pdf = np.sqrt(1 / (2 * np.pi))
                cdf = 1
            else:
                beta = np.sqrt(sigma[0,0]-2*sigma[0,1]+sigma[1,1])
                u = g / beta
                pdf = norm.pdf(u)
                cdf = norm.cdf(u)
            ei_diff = beta * pdf + g * cdf

        self.ei_diff = np.append(self.ei_diff, ei_diff)
        self.KL = np.append(self.KL, np.sqrt(0.5*KL))
        self.kappa = np.append(self.kappa, kappa)
        self.delta_f = np.append(self.delta_f, delta_f)
        self.seq_values = np.append(self.seq_values, delta_f + ei_diff + kappa * np.sqrt(0.5*KL))
        if noise_var is not None:
            x_list = np.concatenate([X[index][None, :], X[-1][None, :]], axis=0)
            mu, sigma = bo_old.model.model.predict_noiseless(x_list, full_cov=True)
            accuracy = 1/noise_var
            c = np.sqrt(-2*np.log(self.delta))

            threshold1 = np.sqrt(sigma[0][0])*np.sqrt(sigma[1][1])*c/(np.sqrt(accuracy)*(sigma[1][1]+noise_var))
            threshold2 = (kappa/2)*np.sqrt(sigma[1][1])*c/(np.sqrt(accuracy)*(sigma[1][1]+noise_var))
            self.threshold1 = np.append(self.threshold1, threshold1)
            self.threshold2 = np.append(self.threshold2, threshold2)
            threshold = threshold1 + threshold2
        self.seq_thresholds = np.append(self.seq_thresholds, threshold)
        if self.seq_values[-1] <= self.seq_thresholds[-1] and not self.stop_flags and self.start_timing < current_time:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, X, y, regret, current_time, **kwargs):
        if not self.stop_flags:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True


class PICriterion(BaseCriterion):
    def __init__(self, budget, start_timing=10, threshold=0.01):
        super(PICriterion, self).__init__("PI",budget,start_timing)
        self.threshold = threshold
        self.threshold_list = np.empty(0, float)
        self.seq_values = np.empty(0, float)

    def set_param(self, param_dim, candidate_param):
        self.param_dim = param_dim
        self.candidate_param = candidate_param.copy()

    def check_threshold(self, bo, X, y, regret, current_time, **kwargs):
        jitter = 0.01
        m, s = bo.model.predict(self.candidate_param)
        fmin = bo.model.get_fmin()
        phi, Phi, u = get_quantiles(jitter, fmin, m, s)
        index = (s * (u * Phi + phi)).argmax()
        self.seq_values = np.append(self.seq_values, Phi[index])
        self.seq_thresholds = np.append(self.seq_thresholds, self.threshold)
        if self.seq_values[-1] <= self.seq_thresholds[-1] and not self.stop_flags and self.start_timing < current_time:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, X, y, regret, current_time, **kwargs):
        if not self.stop_flags:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True


class EICriterion(BaseCriterion):
    def __init__(self,budget,start_timing=10,rate=0.1):
        super(EICriterion, self).__init__("EI",budget,start_timing)
        self.rate = rate
        self.seq_thresholds = np.empty(0, float)
        self.seq_values = np.empty(0, float)
        self.config_name = "-med"

    def set_param(self, param_dim, candidate_param):
        self.param_dim = param_dim
        self.candidate_param = candidate_param.copy()

    def check_threshold(self, bo, X, y, regret, current_time,threshold, **kwargs):
        jitter = 0.01
        m, s = bo.model.predict(self.candidate_param)
        fmin = bo.model.get_fmin()
        phi, Phi, u = get_quantiles(jitter, fmin, m, s)
        ei = (s * (u * Phi + phi)).max()
        self.seq_values = np.append(self.seq_values, ei)
        self.seq_thresholds = np.append(self.seq_thresholds, threshold)
        if self.seq_values[-1] <= self.seq_thresholds[-1] and not self.stop_flags and self.start_timing < current_time:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, X, y, regret, current_time, **kwargs):
        if not self.stop_flags:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True


class RegretCriterion(BaseCriterion):
    def __init__(self,budget,delta=0.1,pool_size=1000,start_timing=10, rate=0.1, threshold_type="med"):
        super(RegretCriterion, self).__init__("SR",budget,start_timing)
        self.rate = rate
        self.seq_thresholds = np.empty(0, float)
        self.seq_values = np.empty(0, float)
        self.delta = delta
        self.pool_size = pool_size
        self.threshold_type = threshold_type

    def set_param(self, param_dim, candidate_param):
        self.param_dim = param_dim
        self.config_name = r"-{}".format(self.threshold_type)
        self.candidate_param = candidate_param.copy()

    def calc_UCB(self,bo,X,sqrt_beta):
        mu_tr, var_tr = bo.model.model.predict_noiseless(X, full_cov=False)
        std_tr = np.sqrt(var_tr)
        return mu_tr+sqrt_beta*std_tr

    def calc_LCB(self,bo,X,sqrt_beta):
        candidate_param = np.concatenate([X, self.candidate_param], axis=0)
        mu_c, var_c = bo.model.model.predict_noiseless(candidate_param, full_cov=False)
        std_c = np.sqrt(var_c)
        return mu_c-sqrt_beta*std_c

    def check_threshold(self,bo, X, y, regret, current_time,threshold, **kwargs):
        sqrt_beta = np.sqrt(2 * np.log(self.param_dim * X.shape[0] ** 2 * np.pi ** 2 / (6 * self.delta)))
        UCB = self.calc_UCB(bo, X, sqrt_beta)
        LCB = self.calc_LCB(bo, X, sqrt_beta)
        self.seq_values = np.append(self.seq_values, np.min(UCB)-np.min(LCB))
        self.seq_thresholds = np.append(self.seq_thresholds, threshold)
        if self.seq_values[-1] <= self.seq_thresholds[-1] and not self.stop_flags and self.start_timing < current_time:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True

    def set_budget_timing(self, X, y, regret, current_time, **kwargs):
        if not self.stop_flags:
            self.stop_timings = current_time
            sc_index = np.argmin(y[:self.stop_timings])
            self.x_opt = X[sc_index]
            self.y_opt = y[sc_index]
            self.regret = regret
            if "test_regret" in kwargs:
                self.test_regret = kwargs["test_regret"]
            print("{} : {}".format(self.name+self.config_name, current_time))
            self.stop_flags = True
