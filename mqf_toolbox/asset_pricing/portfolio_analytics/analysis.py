import numpy as np
import pandas as pd

from warnings import warn

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from .classes import APFrame


def _prep_port_rf(apf: APFrame, risk_free_rate=None, include_bm=False):
    port_arr = apf.portfolio_df.values
    rf = risk_free_rate if risk_free_rate is not None else apf.risk_free_rate

    if include_bm:
        port_arr = np.hstack([port_arr, apf.benchmark_series.values.reshape(-1,1)])

    if type(rf) in [float, type(None)]:
        value = 0 if rf is None else rf
        rf_vector = np.ones(shape=(port_arr.shape[0], 1)) * value
    else:
        rf_vector = np.array(rf).reshape(-1, 1)
        assert rf_vector.shape == (port_arr.shape[0], 1), 'Wrong shape for risk free rate!'

    instrument_list = apf.portfolio_df.columns.to_list()
    if include_bm:
        instrument_list.append(apf.benchmark_series.name)

    return port_arr, rf_vector, instrument_list


def _ols_mat(X, y, constant=True):
    if constant:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    XpX = (X.T @ X)
    XpX_inv = np.linalg.inv(XpX)
    coef = XpX_inv @ X.T @ y

    return coef


def calculate_alpha_beta(apf: APFrame, risk_free_rate=None, include_bm=False):
    port_arr, rf_vector, instrument_list = _prep_port_rf(apf, risk_free_rate, include_bm)
    market_arr = apf.benchmark_series.values[:, np.newaxis]

    X = market_arr - rf_vector
    y = port_arr - rf_vector
    coef = _ols_mat(X, y, constant=True)

    coef_df = pd.DataFrame(coef.T,
                           index=instrument_list,
                           columns=['alpha (Intercept Coefficient)',
                                    'beta (Slope Coefficient)'])

    return coef_df


def calculate_alpha_beta_statsmodels(apf: APFrame, risk_free_rate=None, include_bm=False):
    port_arr, rf_vector, instrument_list = _prep_port_rf(apf, risk_free_rate, include_bm)
    market_arr = apf.benchmark_series.values.reshape(-1, 1)

    ab_dict = {}

    for i, port_ret in enumerate(port_arr.T):
        port_premium = port_ret.reshape(-1, 1) - rf_vector
        market_premium = add_constant(market_arr - rf_vector)

        model = OLS(port_premium, market_premium)
        results = model.fit()

        ab_dict[instrument_list[i]] = {
            'alpha (Intercept Coefficient)': results.params[0],
            'beta (Slope Coefficient)': results.params[1]
        }

    result_df = pd.DataFrame(ab_dict).T
    return result_df


def calculate_alpha_beta_multifactor(apf: APFrame, risk_data: pd.DataFrame = None, risk_free_rate=None,
                                     include_bm=False):
    if risk_data is not None:
        warn("Don't forget to remove risk free rate from market return!",
             stacklevel=2)

        risk_factors = risk_data.values
        risk_factors_names = risk_data.columns.to_list()
    else:
        risk_factors = apf.risk_factors_and_bm.values
        risk_factors_names = apf.risk_factors_and_bm.columns.to_list()

    port_arr, rf_vector, instrument_list = _prep_port_rf(apf, risk_free_rate, include_bm)

    risk_factors_count = len(risk_factors_names)

    assert port_arr.shape[0] == risk_factors.shape[0], 'Number of rows does not match!'

    X = risk_factors
    y = port_arr - rf_vector
    coef = _ols_mat(X, y, constant=True)

    columns_list = [str(risk_factors_count) + ' factor(s) alpha']
    columns_list += [n + ' beta' for n in risk_factors_names]

    coef_df = pd.DataFrame(coef.T,
                           index=instrument_list,
                           columns=columns_list)

    return coef_df


def calculate_sml(apf: APFrame, include_bm=True):
    sml_df = calculate_alpha_beta(apf, include_bm=include_bm)
    expected_ret = apf.portfolio_df.mean()
    if include_bm:
        expected_ret[apf.benchmark_series.name] = apf.benchmark_series.mean()
    sml_df['Expected Return'] = expected_ret

    results = OLS(sml_df['Expected Return'], add_constant(sml_df['beta (Slope Coefficient)'])).fit()
    alpha_sml, beta_sml = results.params

    return alpha_sml, beta_sml


def get_sml_function(apf: APFrame, include_bm=True):
    constant_sml, slope_sml = calculate_sml(apf, include_bm=include_bm)

    def sml_function(beta):
        return constant_sml + slope_sml * beta

    return sml_function


def calculate_sharpe(apf: APFrame, risk_free_rate=None, include_bm=False,
                     ddof=1):
    port_arr, rf_vector, instrument_list = _prep_port_rf(apf, risk_free_rate, include_bm)

    premium = port_arr - rf_vector

    ri_rf = premium.mean(0)
    ri_rf_std = premium.std(0, ddof=ddof)

    sharpe_i = pd.Series(ri_rf / ri_rf_std, index=instrument_list)
    sharpe_i.name = 'Sharpe Ratio'
    return sharpe_i


def calculate_treynor(apf: APFrame, risk_free_rate=None, include_bm=False):
    port_arr, rf_vector, instrument_list = _prep_port_rf(apf, risk_free_rate, include_bm)

    premium = port_arr - rf_vector
    ri_rf = premium.mean(0)

    beta = calculate_alpha_beta(apf, risk_free_rate, include_bm).iloc[:, 1].values

    treynor = pd.Series(ri_rf/beta, index=instrument_list)
    treynor.name = 'Treynor Ratio'
    return treynor


def calculate_jensen_alpha(apf: APFrame, risk_free_rate=None, include_bm=False):
    alpha = calculate_alpha_beta(apf, risk_free_rate, include_bm).iloc[:, 0]
    alpha.name = "Jensen's Alpha"
    return alpha


def calculate_market_beta(apf: APFrame, risk_free_rate=None, include_bm=False):
    beta = calculate_alpha_beta(apf, risk_free_rate, include_bm).iloc[:, 1]
    beta.name = "Market Beta"
    return beta


def calculate_sortino(apf: APFrame, target=None, include_bm=False):
    port_arr, target, instrument_list = _prep_port_rf(apf, target, include_bm)

    target = np.array(target).reshape(-1, 1)
    if target.shape == (1, 1):
        target = target[0][0]
    ri_rt = (port_arr - target).mean(0)
    semi_d = (((port_arr - target).clip(max=0) ** 2).sum(0) / port_arr.shape[0]) ** 0.5

    sortino = pd.Series(ri_rt / semi_d, index=instrument_list, name='Sortino Ratio')
    return sortino


def calculate_5_metrics(apf: APFrame, risk_data: pd.DataFrame = None,
                        risk_free_rate=None, target_rate=None,
                        include_bm=False):
    sharpe = calculate_sharpe(apf, risk_free_rate, include_bm)
    treynor = calculate_treynor(apf, risk_free_rate, include_bm)
    alpha_beta = calculate_alpha_beta(apf, risk_free_rate, include_bm)
    target_rate = risk_free_rate if target_rate is None else target_rate
    sortino = calculate_sortino(apf, target_rate, include_bm)
    alpha = alpha_beta.iloc[:, 0]
    alpha.name = "Jensen's Alpha"
    beta = alpha_beta.iloc[:, 1]
    beta.name = "Market Beta"

    multi_factor_alpha_beta = calculate_alpha_beta_multifactor(apf, risk_data, risk_free_rate, include_bm)
    multi_alpha = multi_factor_alpha_beta.iloc[:, 0]

    return pd.concat([sharpe, sortino, treynor, alpha, multi_alpha], axis=1)
