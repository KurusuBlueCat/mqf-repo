import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Union
from .classes import APFrame, EFCalc
from .analysis import calculate_alpha_beta, get_sml_function


def plot_eff(apf: APFrame, sqrt=False, mode='ef',
             linspace_count=100, y_increment=0.1,
             ylim: Union[Tuple, None] = None,
             color='tab:blue'):
    """
    Handy plotting script for plotting efficient frontier of a given arrays of
    assets returns

    Parameters
    ----------
    apf: APFrame
        QF Frame object containing portfolio of returns
    sqrt: bool
        switch between returning variance (False) or
        volatility (True)
    linspace_count: int = 100
        number of dots for each side of the frontier
        the higher, the smoother the line
    y_increment: float = 0.1
        increment of y-label
    color: str = 'blue'
        pyplot color arg
    figsize: tuple[int, int] = (10,10)
        figure size for pyplot

    Returns
    -------
    None
    """

    assert mode in ['ef', 'te'], "This mode doesn't exist!"
    efc: EFCalc = apf.ef if mode == 'ef' else apf.te

    var_str = 'Std.' if sqrt else 'Var'
    if var_str == 'Std.':
        var_str = 'Std.' if mode == 'ef' else 'Tracking Error'

    ret_str = 'Portfolio Return (%)' if mode == 'ef' else 'Portfolio Benchmark Deviation (%)'

    # Calculation
    mv, mv_y = efc.mv, efc.mv_y
    min_x = mv ** 0.5 if sqrt else mv

    eff = efc.get_ef_function(sqrt)

    if ylim is None:
        upper = mv_y * 2
        lower = 0
    else:
        upper = ylim[1]
        lower = ylim[0]

    return_values_upper = np.linspace(mv_y, upper, linspace_count)  # we can adjust plot range here
    return_values_lower = np.linspace(mv_y, lower, linspace_count)

    variance_values_pos = eff(return_values_upper)

    # Plotting
    plt.plot(variance_values_pos, return_values_upper, color=color, label='Efficient Frontier')
    plt.plot(variance_values_pos, return_values_lower, '--', color=color, label='Inefficient Frontier')
    # plt.axhline(mv_y, ls=':', color=color, label='Minimum ' + var_str + ' Returns')
    plt.axhline(mv_y, ls=':', color=color)
    plt.annotate((round(min_x, 3), round(mv_y, 3)), xy=(min_x, mv_y), size=12)
    plt.plot(min_x, mv_y, 'o', label='Global Minimum ' + var_str + ' Portfolio')
    plt.title('Efficient Frontier')
    plt.ylabel(ret_str)
    plt.xlabel('Portfolio ' + var_str)

    yticks = np.arange(lower, np.round(upper, 1) + y_increment, y_increment)

    plt.yticks(yticks)
    plt.legend()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid()


def plot_tangent(apf, tangency_returns, tangency=True, orthogonal=False,
                 mode='ef', max_x=2,
                 tangency_color='tab:green', orthogonal_color='tab:orange'):
    """
    Handy plotting script for plotting tangent at a given return

    Can also plot the corresponding orthogonal portfolio tangent

    Parameters
    ----------
    apf: APFrame
        QF Frame object containing portfolio of returns
    tangency_returns: float
        return of the portfolio we are finding the tangent of
    tangency: bool = True
        Set to 'False' to not plot the orthogonal portfolio
    orthogonal: bool = False
        Set to 'True' to also plot the orthogonal portfolio

    Returns
    -------
    None
    """

    assert mode in ['ef', 'te'], "This mode doesn't exist!"
    efc: EFCalc = apf.ef if mode == 'ef' else apf.te

    get_sigma = efc.get_ef_function(sqrt=True)
    max_std = get_sigma(max_x * efc.mv_y)
    std_space = np.linspace(0, max_std)

    if tangency:
        tangent_1_f = efc.get_tangency_portfolio_function(tangency_returns)

        tangent_1 = tangent_1_f(std_space)
        label = 'Tangent line'

        x_plot, y_plot = get_sigma(tangency_returns), tangency_returns

        plt.plot(std_space, tangent_1, label=label, color=tangency_color)
        plt.annotate((round(x_plot, 3), round(y_plot, 3)), xy=(x_plot, y_plot), size=12)
        plt.plot(x_plot, y_plot, 'o', color=tangency_color, label='Tangency Portfolio')

        if orthogonal:
            plt.hlines(tangency_returns,
                       0, get_sigma(tangency_returns),
                       color=tangency_color,
                       ls='--')
        else:
            ort_y = efc.calculate_ort_portfolio_returns(tangency_returns)
            plt.annotate((0, round(ort_y, 3)), xy=(0, ort_y), size=12)
            plt.plot(0, ort_y, 'o', color=orthogonal_color, label='Orthogonal Returns')

    if orthogonal:
        tangent_2_f = efc.get_orthogonal_portfolio_function(tangency_returns)
        tangent_2 = tangent_2_f(std_space)
        orthogonal_returns = efc.calculate_ort_portfolio_returns(tangency_returns)

        x_plot, y_plot = get_sigma(orthogonal_returns), orthogonal_returns

        label_2 = 'Orthogonal Tangent Line'

        plt.plot(std_space, tangent_2, label=label_2, color=orthogonal_color)
        plt.annotate((round(x_plot, 3), round(y_plot, 3)), xy=(x_plot, y_plot), size=12)
        plt.plot(x_plot, y_plot, 'o', color=orthogonal_color, label='Orthogonal Portfolio')

        if tangency:
            plt.hlines(orthogonal_returns,
                       0, get_sigma(orthogonal_returns),
                       color=orthogonal_color,
                       ls='-.')
        else:
            tan_y = tangency_returns
            plt.annotate((0, round(tan_y, 3)), xy=(0, tan_y), size=12)
            plt.plot(0, tan_y, 'o', color=tangency_color, label='Tangency Returns')




    plt.legend()


def plot_sml(apf: APFrame,
             plot_range=np.linspace(0, 2, 10),
             bm_name='default',
             security_name='default',
             annotate=False):
    sml_df = calculate_alpha_beta(apf, include_bm=True)
    instrument_count = sml_df.shape[0] - 1
    sml_func = get_sml_function(apf, include_bm=True)
    expected_ret = apf.portfolio_df.mean()
    expected_ret[apf.benchmark_series.name] = apf.benchmark_series.mean()
    sml_df['Expected Return'] = expected_ret

    if bm_name == 'default':
        bm_label = apf.benchmark_series.name
    else:
        bm_label = bm_name

    plt.scatter(
        sml_df.iloc[-1:]['beta (Slope Coefficient)'],
        sml_df.iloc[-1:]['Expected Return'],
        color='tab:orange',
        label=bm_label)

    if security_name == 'default':
        security_label = 'Portfolios'
    else:
        security_label = security_name

    plt.scatter(
        sml_df.iloc[:-1]['beta (Slope Coefficient)'],
        sml_df.iloc[:-1]['Expected Return'],
        color='tab:blue',
        label=security_label)

    plt.legend()

    sml_x = plot_range
    sml_y = sml_func(sml_x)

    plt.plot(sml_x, sml_y)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.ylabel('Expected Return(%)')
    plt.xlabel('Beta')

    plt.grid()

    if annotate:
        for industry, (a, b, r) in sml_df.iterrows():
            plt.annotate(industry, (b + 0.01, r + 0.01))

    plt.title('Security Market Line from '
              + str(instrument_count)
              + ' instrument(s) + market returns')


def plots_metrics(metrics_df, figsize=(10, 16)):
    color = list(mcolors.TABLEAU_COLORS.keys())
    color_count = len(color)

    fig, ax = plt.subplots(len(metrics_df.columns), figsize=figsize)

    for i, (m, s) in enumerate(metrics_df.iteritems()):
        ax[i].bar(s.index, s, label=m, color=color[i % color_count])
        ax[i].set_ylabel(m)
        ax[i].axhline(0, color='gray', ls='--')
        ax[i].grid()
        ax[i].legend()


def plot_montecarlo_portfolio(returns_df, inverse=True, points=100000):
    r_t = returns_df.values[:, :, np.newaxis]
    w_t_rand = np.random.rand(1, r_t.shape[1], points)

    if inverse:
        w_t_rand = 1/w_t_rand

    w_t_rand = w_t_rand / w_t_rand.sum(1).reshape(1, 1, points)

    expected_returns = (r_t * w_t_rand).sum(1).mean(0)
    volatility = (r_t * w_t_rand).sum(1).std(0)
    y_inv = expected_returns
    x_inv = volatility

    label = 'Simulated Portfolio'
    label = label+' inverse' if inverse else label

    plt.scatter(x_inv, y_inv, 2, label=label)
    plt.legend()


def plot_montecarlo_portfolio_apf(apf: APFrame, mode='ef',
                                  inverse=True, points=100000):

    assert mode in ['ef', 'te'], 'Mode does not exist!'

    returns_df = apf.portfolio_df if mode == 'ef' else apf.deviations_df
    plot_montecarlo_portfolio(returns_df, inverse, points)

    if mode == 'ef':
        plt.ylabel('Portfolio Returns (%)')
        plt.xlabel('Portfolio Std. (%)')

    if mode == 'te':
        plt.ylabel('Portfolio Deviations (%)')
        plt.xlabel('Portfolio Tracking Error (%)')
