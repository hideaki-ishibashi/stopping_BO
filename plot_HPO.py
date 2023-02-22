import matplotlib.pylab as plt
from model.stopping_criteria import *
from utils import draw_result
import os


def main():
    n_iterate = 10
    budget = 500
    fontsize = 20
    ticksize = 15
    acq_func_name = "LCB"
    # classification : "logistic","svc","rfc". regression : "ridge","svr","rfr"
    model_names = ["ridge", "svr", "rfr", "logistic","svc","rfc"]
    data_names = {"ridge": ["gas_turbine", "power_plant", "protein"], "svr": ["gas_turbine", "power_plant", "protein"],
                  "rfr": ["gas_turbine", "power_plant", "protein"], "logistic": ["skin", "HTRU2", "electrical_grid_stability"],
                  "svc": ["skin", "HTRU2", "electrical_grid_stability"], "rfc": ["skin", "HTRU2", "electrical_grid_stability"]}
    displayLegends = [True, False, False, False, False, False]

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    for t, model_name in enumerate(model_names):
        displayLegend = displayLegends[t]
        for data_name in data_names[model_name]:
            dir = "result/HPO_experiments/" + data_name + "/" + model_name + "/"
            os.makedirs(dir, exist_ok=True)

            fig1 = plt.figure(1,[5,5])
            ax1=plt.subplot(111)
            ax1.cla()
            fig2 = plt.figure(2,[5,5])
            ax2=plt.subplot(111)
            ax2.cla()

            # load dataset and results
            rmses = np.loadtxt(dir + "discretized_rmse.txt")
            hpo = utils.deserialize(dir + "hpo_{}_{}.dat".format(acq_func_name, 0))
            regret_init = np.zeros(n_iterate)
            regret_cv = np.zeros((len(hpo.stopping_criteria),n_iterate))
            regret_time = np.zeros((len(hpo.stopping_criteria),n_iterate))
            ryc_test = np.zeros((len(hpo.stopping_criteria),n_iterate))
            rtc = np.zeros((len(hpo.stopping_criteria),n_iterate))
            reg = np.zeros((len(hpo.stopping_criteria),n_iterate))
            for e in range(n_iterate):
                hpo = utils.deserialize(dir+"hpo_{}_{}.dat".format(acq_func_name, e))
                regret_init[e] = hpo.regret[0]
                for i, sc in enumerate(hpo.stopping_criteria):
                    regret_cv[i, e] = hpo.regret[sc.stop_timings-1] / (rmses.max() - rmses.min())
                    ryc_test[i, e] = (hpo.test_regret[-1] - hpo.test_regret[sc.stop_timings-1])/np.max([hpo.test_regret[-1], hpo.test_regret[sc.stop_timings-1], 1e-10])
                    rtc[i, e] = (budget - sc.stop_timings)/budget
                    regret_time[i, e] = sc.stop_timings
                    reg[i, e] = sc.regret
            normalized_regret = (rmses - rmses.min())/(rmses.max()-rmses.min())
            sorted_normalized_regret = np.sort(normalized_regret)

            # draw RYC and RTC of each stopping criterion
            draw_result.draw_ryc_rtc(ax1, ryc_test, rtc, hpo.stopping_criteria, fontsize, ticksize,
                                     isLegend=displayLegend)
            fig1.savefig(dir + "{}_{}_ryc_rtc_test_{}.pdf".format(data_name,acq_func_name, model_name),
                         bbox_inches='tight', pad_inches=0)
            # draw normalized simple regret and stopping timing of each stopping criterion
            draw_result.draw_reg_time(ax2, regret_cv, regret_time, hpo.stopping_criteria, budget, fontsize, ticksize,
                                      sorted_normalized_regret, isLegend=displayLegend)
            fig2.savefig(dir + "{}_{}_regret_{}.pdf".format(data_name,acq_func_name, model_name), bbox_inches='tight',
                         pad_inches=0)
            plt.pause(0.01)


if __name__ == "__main__":
    main()
