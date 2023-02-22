import matplotlib.pylab as plt


def draw_objective_function_2d(ax, grid, objective_function, discovered_points=None, criteria=None, sample=None, best_points=None,budget_points=None,fontsize=15,isLegend=False):
    node_size = grid.shape[0]
    contour = objective_function.reshape(node_size,node_size)
    ax.set_xlabel(r"$\theta_1$", fontsize=fontsize)
    ax.set_ylabel(r"$\theta_2$", fontsize=fontsize)
    ax.contour(grid[:,:,0], grid[:,:,1], contour,levels=100, cmap="rainbow",zorder=1)
    if sample is not None:
        ax.scatter(sample[:,0],sample[:,1],c="k",edgecolor="k",marker="x",label="Explored data",zorder=2)
    if best_points is not None:
        ax.scatter(best_points[:,0],best_points[:,1],c='r',marker="*",s=200,edgecolor="k",label="Optimal point",zorder=4)
    if budget_points is not None:
        ax.scatter(budget_points[:,0],budget_points[:,1],c='yellow',edgecolor="k",label="Max budget",zorder=3)
    if discovered_points is not None:
        colormap = plt.get_cmap("rainbow", discovered_points.shape[0])
        for i, sc in enumerate(criteria):
            if r"f^\ast" in sc.config_name:
                sc.config_name = r"-{}-$f^\ast$".format(sc.threshold_type)
            ax.scatter(discovered_points[i, :, 0], discovered_points[i, :, 1],c=(colormap(i),),edgecolor="k",label=sc.name+sc.config_name,zorder=3)
    if isLegend:
       ax.legend(fontsize=fontsize)


def draw_ryc_rtc(ax, ryc, rtc, stopping_criteria, fontsize, ticksize, isLegend=False, fix_lim=True):
    ax.cla()
    ax.tick_params(labelsize=ticksize)
    if fix_lim:
        ax.set_xlim(0, 1)
    ax.set_xlabel("RTC", fontsize=fontsize)
    ax.set_ylabel("RYC", fontsize=fontsize)
    colormap = plt.get_cmap("rainbow", len(stopping_criteria))
    for i, sc in enumerate(stopping_criteria):
        ax.errorbar(rtc[i].mean(), ryc[i].mean(), xerr=rtc[i].std(), yerr=ryc[i].std(), capsize=5, fmt='o', marker="s", markersize=5, markeredgecolor="k",
                     color=colormap(i))
        ax.scatter(rtc[i],ryc[i],c=(colormap(i),),label=sc.name+sc.config_name,alpha=0.5)
    if isLegend:
        ax.legend(fontsize=15)


def draw_reg_time(ax, reg, times, stopping_criteria, budget, fontsize, ticksize, normalized_regret=None, isLegend=False,fix_lim=True):
    ax.cla()
    ax.tick_params(labelsize=ticksize)
    if fix_lim:
        ax.set_xlim(-20, budget+10)
    ax.set_xlabel("Number of evaluations", fontsize=fontsize)
    ax.set_ylabel("Normalized simple regret", fontsize=fontsize)
    colormap = plt.get_cmap("rainbow", len(stopping_criteria))
    if normalized_regret is not None:
        ax.axhline(y=normalized_regret[9], c="k", label="Top10", ls="dashdot")
    for i, sc in enumerate(stopping_criteria):
        if r"f^\ast" in sc.config_name:
            sc.config_name = r"-{}-$f^\ast$".format(sc.threshold_type)
        ax.errorbar(times[i].mean(), reg[i].mean(), xerr=times[i].std(), yerr=reg[i].std(), capsize=5, fmt='o', marker="s", markersize=5, markeredgecolor="k",
                     color=colormap(i))
        ax.scatter(times[i],reg[i],c=(colormap(i),),label=sc.name+sc.config_name,alpha=0.5)
    if isLegend:
        ax.legend(fontsize=15)


def draw_criterion_value(ax, criteria, time_array, fontsize, displayLegend):
    I = time_array.shape[0]
    ax.set_xlim(0, I)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-12, top=1e3)
    rainbow = plt.get_cmap("rainbow", len(criteria))
    for i, sc in enumerate(criteria):
        if r"f^\ast" in sc.config_name:
            sc.config_name = r"-{}-$f^\ast$".format(sc.threshold_type)
        ax.plot(time_array, sc.seq_values, c=rainbow(i), label=sc.name + sc.config_name)
        if sc.name == "Ours":
            if sc.threshold_type == "auto":
                ax.plot(time_array, sc.seq_thresholds, c="k", label="threshold")

    ax.set_xlabel("Number of evaluations", fontsize=fontsize)
    ax.set_ylabel("Value of stopping criterion", fontsize=fontsize)
    if displayLegend:
        ax.legend(fontsize=fontsize)


def draw_sec_hyper_param(ax, hyper_param, param_name, fontsize, color, displayLegend):
    ax.set_xlim(0, len(hyper_param))
    print(hyper_param)
    # ax.set_yscale('log')
    # ax.set_ylim(bottom=1e-12, top=1e3)
    # ax.plot(range(len(hyper_param)), hyper_param, c=color, label=param_name)
    ax.plot(range(len(hyper_param)), hyper_param, c=color)
    ax.set_xlabel("Number of evaluations", fontsize=fontsize)
    ax.set_ylabel(param_name, fontsize=fontsize)
    # if displayLegend:
    #     ax.legend(fontsize=fontsize)


def draw_decomposed_proposed_value(ax, criteria, times, fontsize, displayLegend):
    I = times.shape[0]
    ax.set_xlim(0, I)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-12, top=1e3)
    for i, sc in enumerate(criteria):
        if sc.config_name == "-med":
            if sc.delta_f is not None:
                ax.plot(times, sc.delta_f, c="b", label=r"$|\Delta \mu^\ast_t|$")
            if sc.ei_diff is not None:
                ax.plot(times, sc.ei_diff, c="r", label=r"$v(\phi(g)+g\Phi(g))$")
            if sc.kappa is not None:
                ax.plot(times, sc.kappa*sc.KL, c="g", label=r"$\kappa_t\sqrt{\frac{1}{2}D_{\rm KL}[p_t(f)||p_{t-1}(f)]}$")
            if sc.kappa is not None:
                ax.plot(times, sc.kappa, c="purple", label=r"$\kappa_t$")
            if sc.KL is not None:
                ax.plot(times, sc.KL, c="cyan", label=r"$\sqrt{\frac{1}{2}D_{\rm KL}[p_t(f)||p_{t-1}(f)]}$")

    ax.set_xlabel("Number of evaluations", fontsize=fontsize)
    ax.set_ylabel("Value of the proposed criterion", fontsize=fontsize)
    if displayLegend:
        ax.legend(fontsize=fontsize)


def draw_error(ax,criteria, times, val_opts, test_opts=None, fontsize=15, scale="log"):
    I = times.shape[0]
    ax.cla()
    ax.set_xlim(0, I)
    if scale == "log":
        ax.set_yscale('log')
    current_time = len(criteria[0]["criterion"].seq_values)
    ax.plot(times[:current_time], val_opts[:current_time],c='k', label="Regret of validation error")
    if test_opts is not None:
        ax.plot(times, test_opts, c='orange', label="Regret of test error")
    for i,criterion in enumerate(criteria):
        ax.axvline(criterion["criterion"].stop_timings, 0, 1e5, linestyle='dashed', c=criterion["color"])

    ax.set_xlabel("data size", fontsize=fontsize)
    ax.set_ylabel("simple regret", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
