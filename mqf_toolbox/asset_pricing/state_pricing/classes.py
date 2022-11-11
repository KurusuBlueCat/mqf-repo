import pandas as pd
import numpy as np


class StatePricing:
    _rf_str = 'rf'

    def __init__(self, init_price_series, state_payoff_df, risk_free_rate=10, physical_prob_series=None):
        self.risk_free_rate = risk_free_rate

        price_s = init_price_series.copy()
        price_s = pd.concat([pd.Series(1 / self._rf_factor, index=[self._rf_str]), price_s])

        self.init_price_series = price_s

        payoff_df = state_payoff_df.copy()
        payoff_df = pd.concat(
            [pd.DataFrame([[1] * payoff_df.shape[1]], index=[self._rf_str], columns=payoff_df.columns),
             payoff_df])

        self.state_payoff_df = payoff_df

        self.physical_prob_series = physical_prob_series

    @property
    def risk_free_rate(self):
        return self._rf_percent

    @risk_free_rate.setter
    def risk_free_rate(self, value):
        self._rf_percent = value
        self._rf_factor = 1 + value / 100

    @property
    def physical_prob_series(self):
        if self._prob_s is None:
            raise AttributeError('physical_prob_series was not set!')
        return self._prob_s

    @physical_prob_series.setter
    def physical_prob_series(self, value):
        if value is not None:
            assert all(value.index == self.state_payoff_df.columns)

        self._prob_s = value

    @property
    def state_price(self):
        state_price = self.init_price_series.values @ np.linalg.inv(self.state_payoff_df.values.T)
        state_price = pd.Series(state_price, index=self.state_payoff_df.columns, name='state price')
        return state_price

    @property
    def risk_neutral_vector(self):
        state_price = self.state_price
        risk_neutral = state_price * self._rf_factor
        risk_neutral = pd.Series(risk_neutral, index=state_price.index, name='risk neutral vector')
        return risk_neutral

    @property
    def pricing_kernel(self):
        pricing_kernel = self.state_price / self.physical_prob_series
        pricing_kernel.name = 'pricing kernel'
        return pricing_kernel

    @property
    def radon_nikodym_d(self):
        rn = self._rf_factor * self.pricing_kernel
        rn.name = 'radon nikodym derivative'
        return rn

    def _get_underlying(self):
        underlying_list = self.state_payoff_df.index.to_list()
        underlying_list = [i for i in underlying_list if i != self._rf_str]
        return underlying_list

    def calculate_call_payoff(self, strike, underlying_list=None):
        if underlying_list is None:
            underlying_list = self._get_underlying()

        payoff = self.state_payoff_df.loc[underlying_list].sum() - strike
        payoff = payoff.clip(lower=0)
        return payoff

    def calculate_put_payoff(self, strike, underlying_list=None):
        if underlying_list is None:
            underlying_list = self._get_underlying()

        payoff = strike - self.state_payoff_df.loc[underlying_list].sum()
        payoff = payoff.clip(lower=0)
        return payoff

    def payoff_valuation(self, payoff, risk_neutral=True):
        if risk_neutral:
            return (payoff * self.state_price).sum()
        else:
            return (payoff * self.physical_prob_series / self._rf_factor).sum()
