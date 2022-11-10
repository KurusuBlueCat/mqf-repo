import numpy as np
import pandas as pd
from typing import Union
from .functions import get_greeks, greeks_to_param, get_efficient_frontier_function
from warnings import warn


class EFCalc:
    """
    Small class that calculates efficient frontier related values given
    a dataframe of returns.

    Parameters
    ----------
    returns_df: pd.DataFrame
        A dataframe of instruments returns in percentage values (1 is 1%).

    """

    def __init__(self, returns_df):
        alpha, zeta, delta, cov_inv, mean = get_greeks(returns_df,
                                                       return_inv_r=True)
        mv, var_increment, mv_y = greeks_to_param(alpha,
                                                  zeta,
                                                  delta)

        divisor = zeta * delta - alpha ** 2
        e_v = np.ones(mean.shape)

        base_weight = (zeta * cov_inv @ e_v
                       - alpha * cov_inv @ mean) / divisor
        weight_increment = (delta * cov_inv @ mean
                            - alpha * cov_inv @ e_v) / divisor

        self.alpha = alpha
        self.zeta = zeta
        self.delta = delta
        self.cov_inv = cov_inv
        self.mean = mean
        self.mv = mv
        self.var_increment = var_increment
        self.mv_y = mv_y
        self.base_weight = base_weight
        self.weight_increment = weight_increment

    def get_ef_function(self, sqrt=True):
        """
        Create function for plotting efficient frontier

        Parameters
        ----------
        sqrt: bool
            Set to False to make the output values variance. True for volatility/std.

        Returns
        -------
        ef_function: get_efficient_frontier_function
            calculates variance/std from return value
        """
        return get_efficient_frontier_function(self.mv, self.var_increment, self.mv_y, sqrt)

    def calculate_portfolio_weights(self, returns):
        """
        Calculate the portfolio weight required to achieve the target
        returns with minimum variance. The weight always sum to one.

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        portfolio_weights: np.ndarray
            Arrays of portfolio weights

        """
        return np.array(self.base_weight + self.weight_increment * returns).reshape((-1,))

    def calculate_min_std(self, returns):
        """
        Calculate the minimum standard deviation possible given target returns

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        standard_deviation: float
            standard deviation in percentage unit

        """
        return self.get_ef_function(sqrt=True)(returns)

    def calculate_min_var(self, returns):
        """
        Calculate the minimum variance possible given target returns

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        standard_deviation: float
            standard deviation in percentage unit

        """
        return self.get_ef_function(sqrt=False)(returns)

    def calculate_tangency_portfolio_gradient(self, returns):
        """
        Calculate the gradient of the tangent at a given return level

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        gradient: float
            gradient of the tangent

        """
        zeta = self.zeta
        alpha = self.alpha
        delta = self.delta
        s_1 = self.calculate_min_std(returns)

        return (zeta * delta - alpha ** 2) * s_1 / (delta * (returns - self.mv_y))

    def calculate_ort_portfolio_returns(self, returns):
        """
        Calculate the returns of portfolio that is 'orthogonal' to the
        minimum variance portfolio at a given target returns.

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        returns: float
            orthogonal portfolio returns

        """
        grad = self.calculate_tangency_portfolio_gradient(returns)
        s_1 = self.calculate_min_std(returns)

        return returns - grad * s_1

    def calculate_ort_portfolio_gradient(self, returns):
        """
        Calculate the gradient of the portfolio that is 'orthogonal' to the
        minimum variance portfolio at a given target returns.

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        gradient: float
            orthogonal portfolio gradient

        """
        ort_ret = self.calculate_ort_portfolio_returns(returns)
        return self.calculate_tangency_portfolio_gradient(ort_ret)

    def get_tangency_portfolio_function(self, returns):
        """
        Create a function that plots the tangent of the minimum variance
        portfolio given target returns.

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        tangency_function: Callable

        """
        y_intercept = self.calculate_ort_portfolio_returns(returns)
        grad = self.calculate_tangency_portfolio_gradient(returns)

        def tangency_function(std):
            """
            Function for plotting tangent line.
            X-axis (input) is standard deviation.
            Y-axis (output) is returns.

            (Of course you can plot x on the y and y on the x, but why)

            Parameters
            ----------
            std: float
                standard deviation on the X axis

            Returns
            -------
            returns: float
                returns on the Y axis

            """
            return y_intercept + grad * std

        return tangency_function

    def get_orthogonal_portfolio_function(self, returns):
        """
        Create a function that plots the tangent of the orthogonal
        minimum variance portfolio given target returns.

        Parameters
        ----------
        returns: float
            target returns

        Returns
        -------
        ort_tangency_function: Callable

        """
        ort_ret = self.calculate_ort_portfolio_returns(returns)
        return self.get_tangency_portfolio_function(ort_ret)


class APFrame:
    """
    'Asset-Pricing' frame made just for QF600

    Parameters
    ----------
    portfolio_df: pd.DataFrame
        A DataFrame of nxm, n rows of dates, m columns of securities.
        Values are percentage returns, 1 = 1%.

    benchmark_series: pd.Series, optional
        A series of benchmark returns.
        Should contain the same rows as portfolio_df.

    risk_factor_df: pd.DataFrame, optional
        A DataFrame of nxm, n rows of dates, m columns of risk factors.
        Do not need to include rm-rf, instead plug your rf into risk_free_rate,
        and your rm into benchmark_series.
        Should contain the same rows as portfolio_df.

    risk_free_rate: Union[pd.Series, float], optional
        Either a series or a single constant risk-free-rate.
        If a series, should contain the same rows as portfolio_df.
    """

    def __init__(self, portfolio_df: pd.DataFrame,
                 benchmark_series: pd.Series = None,
                 risk_factor_df: pd.DataFrame = None,
                 risk_free_rate: Union[pd.Series, float, None] = None):
        self._rf_s = None
        self._bm_s = None
        self._risk_df = None
        self.portfolio_df = portfolio_df

        if benchmark_series is not None:
            self.benchmark_series = benchmark_series

        if risk_factor_df is not None:
            self.risk_factor_df = risk_factor_df

        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate

    # property dec port
    @property
    def portfolio_df(self):
        """

        Returns
        -------
        portfolio_df: pd.DataFrame
            A DataFrame of nxm, n rows of dates, m columns of securities.

        """
        return self._p_df

    @portfolio_df.setter
    def portfolio_df(self, value):
        """
        Setter function for portfolio_df.

        Creates an attribute 'ef' which is an instance of EFCalc.
        This instance will be dedicated to calculation of efficient frontier
        analysis.

        Will automatically recalculate everything as well.

        Parameters
        ----------
        value: pd.DataFrame
            A DataFrame of nxm, n rows of dates, m columns of securities.

        """
        self._p_df = value

        self.ef = EFCalc(value)

        # re calc bm and rf related calculations
        if self._bm_s is not None:
            self.benchmark_series = self._bm_s

        if self._risk_df is not None:
            self.risk_factor_df = self._risk_df

        if self._rf_s is not None:
            self.risk_free_rate = self._rf_s

    # bm

    @property
    def benchmark_series(self):
        """
        Will raise attribute error if benchmark_series was not set.

        Returns
        -------
        benchmark_series: pd.Series
            A series of benchmark returns.

        """
        if self._bm_s is None:
            raise AttributeError('benchmark_series was not set!')

        return self._bm_s

    @benchmark_series.setter
    def benchmark_series(self, value):
        """
        Setter function for benchmark_series.

        Will automatically tries to convert 1 columns dataframe to series.

        Will cross-check with existing risk factor to check if any duplicated
        risk factor exists.

        Will also create an attribute 'te', an instance of EFCalc dedicated to
        analysis of efficient frontier based on relative returns/tracking error
        against benchmark.


        Parameters
        ----------
        value: pd.Series
            A series of benchmark returns.

        """
        if value is not None:
            if type(value) is pd.DataFrame:
                assert value.shape[1] == 1, 'Does not accept multiple columns for benchmark!'
                value = value.iloc[:, 0]  # convert to series

            self._bm_s = value

            if self._risk_df is not None:
                self._check_bm_and_risk()

            self.deviations_df = self.portfolio_df - value.values.reshape(-1, 1)

            self.te = EFCalc(self.deviations_df)

            self.bm_tangency_returns = self.te.calculate_ort_portfolio_returns(0)
            self.bm_tangency_sharpe = self.te.calculate_ort_portfolio_gradient(0)

    # risk factor

    @property
    def risk_factor_df(self):
        """

        Returns
        -------
        risk_factor_df: pd.DataFrame
            A DataFrame of nxm, n rows of dates, m columns of risk factors.
            Do not need to include rm-rf, instead plug your rf into risk_free_rate,
            and your rm into benchmark_series.

        """
        if self._risk_df is None:
            raise AttributeError('risk_factor_df was not set!')

        return self._risk_df

    @risk_factor_df.setter
    def risk_factor_df(self, value):
        """
        Setter Function for risk factors.

        Will cross-check with benchmark to identify correlated/identical risk factor

        Parameters
        ----------
        value: pd.DataFrame
            A DataFrame of nxm, n rows of dates, m columns of risk factors.
            Do not need to include rm-rf, instead plug your rf into risk_free_rate,
            and your rm into benchmark_series.

        """
        if value is not None:
            if type(value) is pd.Series:
                value = value.to_frame()

            self._risk_df = value

            if self._bm_s is not None:
                self._check_bm_and_risk()

    def _check_bm_and_risk(self):
        for risk_name, risk in self.risk_factor_df.iteritems():
            corr = np.corrcoef(self.benchmark_series.values, risk.values)[0, 1]
            if corr > 0.7:
                warn(str(risk_name) + ' is highly correlated with benchmark ' + str(self.benchmark_series.name),
                     stacklevel=3)

    # rf

    @property
    def risk_free_rate(self):
        """

        Returns
        -------
        risk_free_rate: pd.Series
            A series of risk_free_rate.

        """
        if self._rf_s is None:
            raise AttributeError('risk_free_rate was not set!')

        return self._rf_s

    @risk_free_rate.setter
    def risk_free_rate(self, value):
        """

        Setter function for risk_free_rate.

        Will convert float into a series with the same index as portfolio.

        Parameters
        ----------
        value: Union[pd.Series, float], optional
            Either a series or a single constant risk-free-rate.
            If a series, should contain the same rows as portfolio_df.

        Returns
        -------

        """
        if value is not None:
            if type(value) is pd.DataFrame:
                assert value.shape[1] == 1, 'Does not accept multiple columns for risk_free_rate!'
                value = value.iloc[:, 0]  # convert to series

            if type(value) is pd.Series:
                self._rf_s = value
                assert self._rf_s.shape[0] == self.portfolio_df.values.shape[0], \
                    'Row number of risk free rate does not match with portfolio_df!'

            else:
                self._rf_s = pd.Series(value, index=self.portfolio_df.index, name='Risk-Free Rate')

            self.rf_tangency_returns = self.ef.calculate_ort_portfolio_returns(self._rf_s.mean())
            self.rf_tangency_sharpe = self.ef.calculate_ort_portfolio_gradient(self._rf_s.mean())
            self.rf_expected_risk_premiums = (self.portfolio_df - self._rf_s.values.reshape(-1, 1)).mean()
            self.rf_expected_risk_premiums.name = 'Expected Premiums'

    # purely port df

    @property
    def expected_returns(self):
        """
        Property for easy access to self.portfolio_df.mean()

        Returns
        -------
        expected_returns: pd.Series
            A series of average returns of the portfolio_df

        """
        ret_series = self.portfolio_df.mean()
        ret_series.name = 'Expected Returns'
        return ret_series

    @property
    def expected_deviations(self):
        """
        Property for easy access to expected deviation from benchmark

        Returns
        -------
        expected_deviations: pd.Series
            A series of average difference of the portfolio_df - self.benchmark_series

        """
        deviations = self.portfolio_df - self.benchmark_series.values.reshape(-1, 1)
        expected_deviations = deviations.mean()
        expected_deviations.name = 'Expected Deviations'
        return expected_deviations

    def calculate_sortino(self, target=None):
        """
        sortino

        Parameters
        ----------
        target: float
            Target return for calculating shortfall

        Returns
        -------
        sortino_ratio: pd.Series
            A series of return above target divided by downside semi deviation

        """
        target = target if target is not None else self.risk_free_rate
        target = np.array(target).reshape(-1, 1)
        if target.shape == (1, 1):
            target = target[0][0]
        ri_rt = (self.portfolio_df - target).mean()
        semi_d = (((self.portfolio_df - target).clip(upper=0) ** 2).sum() / self.portfolio_df.shape[0]) ** 0.5

        sortino = ri_rt / semi_d
        sortino.name = 'Sortino Ratio'
        return sortino

    # use rf

    def cal_function_from_std(self, std):
        """
        cal = capital asset line

        this function is like the result of calling ef.get_orthogonal_portfolio_function
        with a constant risk-free-rate as the target returns.

        Handy for plotting

        Parameters
        ----------
        std: float
            input values of vol

        Returns
        -------
        returns: float
            output values of returns that is on the cal line given vol

        """
        return self.risk_free_rate.mean() + self.rf_tangency_sharpe * std

    def cal_function_from_return(self, returns):
        """
        cal = capital asset line
        inverse of cal_function_from_std

        Handy for plotting

        Parameters
        ----------
        returns: float
            input values of returns

        Returns
        -------
        std: float
            output values of std that is on the cal line given returns

        """

        var = ((returns - self.risk_free_rate.mean()) ** 2) / (self.rf_tangency_sharpe ** 2)
        return var ** 0.5

    def calculate_weights_rf(self, returns):
        """
        Calculate portfolio weights required to achieve target returns.
        Total weights below/above 1 implies investing in risk-free/debt leveraging.

        Sanity Check:
            ef.calculate_portfolio_weights(self.rf_tangency_returns)
            and
            calculate_weights_rf(self.rf_tangency_returns)
            should give identical results.

        Parameters
        ----------
        returns:
            target returns

        Returns
        -------
        portfolio_weights: np.ndarray
            Arrays of portfolio weights

        """
        r_f = self.risk_free_rate.mean()
        lagrange_n = (returns - r_f)
        # there's a sharpe squared here?
        lagrange_d = (self.ef.zeta - 2 * self.ef.alpha * r_f + self.ef.delta * (r_f ** 2))
        lagrange = lagrange_n / lagrange_d
        return (lagrange * self.ef.cov_inv @ self.rf_expected_risk_premiums.values).A.reshape(-1)

    def calculate_sharpe(self, risk_free_rate=None):
        """
        good ol sharpe ratio

        Parameters
        ----------
        risk_free_rate: float, pd.Series, np.ndarray
            risk-free rate for sharpe calculation

        Returns
        -------
        sharpe_ratio: pd.Series:
            a series of sharpe ratio for each instrument

        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        if type(rf) not in [float, int]:
            rf = rf.values.reshape(-1, 1)

        data_i_premium = self.portfolio_df - rf
        ri_rf = data_i_premium.mean()
        ri_rf_std = data_i_premium.std()

        sharpe_i = ri_rf / ri_rf_std
        sharpe_i.name = 'Sharpe Ratio'
        return sharpe_i

    # use all

    @property
    def risk_factors_and_bm(self):
        if (self._risk_df is None) and (self._bm_s is None):
            raise AttributeError('Neither risk_factor_df or benchmark_series were set!')

        risk_free = self._rf_s.values.reshape(-1, 1) if self._rf_s is not None else np.zeros((self._bm_s.shape[0], 1))

        bm_premium: pd.DataFrame = self._bm_s.to_frame() - risk_free if self._bm_s is not None else None
        bm_premium.columns = [self._bm_s.name + ' - rf']
        r = self._risk_df if self._risk_df is not None else None

        return pd.concat([i for i in [bm_premium, r] if i is not None], axis=1)

    def __repr__(self):
        return "APFrame's Portfolio DF:\n" + self.portfolio_df.__repr__()

    def __str__(self):
        return "APFrame's Portfolio DF:\n" + self.portfolio_df.__str__()
