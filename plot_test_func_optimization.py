import matplotlib.pylab as plt
from model.stopping_criteria import *
from utils import utils, bo_utils, draw_result
from get_dataset import test_func
import os

def main():
    # experimental parameter
    budgets = [500, 1000, 500, 1000, 500, 500]  # budget of BO
    acq_func_name = 'LCB'  # acuisition function
    function_names = ["holder_table", "cross_in_tray", "six_hump_camel", "easom", "rosenbrock", "booth"]
    n_iterate = 10

    displayLegends = [True, False, False, False, False, False]
    ticksize = 15
    fontsize = 20

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    fig1 = plt.figure(1, [6, 6])
    ax1 = plt.subplot(111)
    fig2 = plt.figure(2, [6, 6])
    ax2 = plt.subplot(111)
    fig3 = plt.figure(3, [6, 6])
    ax3 = plt.subplot(111)
    fig4 = plt.figure(4, [6, 6])
    ax4 = plt.subplot(111)
    fig5 = plt.figure(5, [6, 6])
    ax5 = plt.subplot(111)
    fig6 = plt.figure(6, [6, 6])
    ax6 = plt.subplot(111)
    for i, function_name in enumerate(function_names):
        budget = budgets[i]
        displayLegend = displayLegends[i]
        dir = "result/BO_test_func/" + function_name + "/"
        os.makedirs(dir, exist_ok=True)

        # get test function's configuration
        x_range, x_min = test_func.get_config(function_name)
        node_size = 1000
        bounds, param_dim, param_range = bo_utils.set_bounds("test_func", x_range)
        grid, node = utils.create_grid(node_size, param_dim, x_range)
        f_min = test_func.function(x_min, x_min, function_name)[0]
        f_max = test_func.function(node, x_min, function_name).max()

        # get discreatized objective function for visualize the result
        node_size = 100
        grid, node = utils.create_grid(node_size, param_dim, x_range)
        objective_function = test_func.function(node, x_min, function_name)

        # draw objective function
        ax1.cla()
        draw_result.draw_objective_function_2d(ax1, grid, objective_function, best_points=x_min, fontsize=fontsize, isLegend=displayLegend)
        fig1.tight_layout()
        fig1.savefig(dir + "objective_function_{}.pdf".format(function_name))
        ax1.cla()

        tfo = utils.deserialize(save_name=dir + "tfo_{}_0.dat".format(acq_func_name))
        time_array = np.arange(budget - 1)
        normalized_simple_regret = np.zeros((len(tfo.stopping_criteria), n_iterate))
        stop_timings = np.zeros((len(tfo.stopping_criteria), n_iterate))
        discovered_points = np.zeros((len(tfo.stopping_criteria), n_iterate, 2))
        for e in range(n_iterate):
            # load results
            tfo = utils.deserialize(save_name=dir + "tfo_{}_{}.dat".format(acq_func_name,e))
            for s, sc in enumerate(tfo.stopping_criteria):
                normalized_simple_regret[s, e] = tfo.regret[sc.stop_timings - 1]/(f_max-f_min)
                stop_timings[s, e] = sc.stop_timings
                discovered_points[s, e] = sc.x_opt
            # draw the explored points
            ax1.cla()
            draw_result.draw_objective_function_2d(ax1, grid, objective_function, sample=tfo.X, fontsize=fontsize,
                                                   isLegend=displayLegend)
            fig1.tight_layout()
            fig1.savefig(dir + "{}_sample_{}_{}.pdf".format(acq_func_name, function_name,e))
            # draw the quantities of each stopping criterion
            ax2.cla()
            draw_result.draw_criterion_value(ax2, tfo.stopping_criteria, time_array, fontsize, displayLegend)
            fig2.tight_layout()
            fig2.savefig(dir + "{}_criteria_value_{}_{}.pdf".format(acq_func_name, function_name, e))

            # draw each term of the amount of the proposed criterion
            ax3.cla()
            draw_result.draw_decomposed_proposed_value(ax3, tfo.stopping_criteria, time_array, fontsize, displayLegend)
            fig3.tight_layout()
            fig3.savefig(dir + "{}_proposed_value_{}_{}.pdf".format(acq_func_name, function_name, e))

            # draw sequence of the hyperparameter of GP
            fig6.tight_layout()
            colors = {"var": "k", "length": "r", "noise_var": "g"}
            for param_name in tfo.history.keys():
                ax6.cla()
                draw_result.draw_sec_hyper_param(ax6, tfo.history[param_name], param_name, fontsize, colors[param_name], displayLegend)
                fig6.savefig(dir + "{}_{}_{}_{}.pdf".format(acq_func_name, param_name, function_name, e))
        # draw each term of the amount of the proposed criterion
        ax4.cla()
        draw_result.draw_objective_function_2d(ax4, grid, objective_function, discovered_points=discovered_points,
                                               criteria=tfo.stopping_criteria, fontsize=fontsize, isLegend=displayLegend)
        fig4.tight_layout()
        fig4.savefig(dir + "{}_objective_function_{}.pdf".format(acq_func_name, function_name))
        # draw normalized simple regret and stopping timing of each stopping criterion
        draw_result.draw_reg_time(ax5, normalized_simple_regret, stop_timings, tfo.stopping_criteria, budget, fontsize,
                                  ticksize, isLegend=displayLegend)
        fig5.tight_layout()
        fig5.savefig(dir + "{}_regret_trials_{}.pdf".format(acq_func_name, function_name))
        plt.pause(0.01)


if __name__ == "__main__":
    main()
