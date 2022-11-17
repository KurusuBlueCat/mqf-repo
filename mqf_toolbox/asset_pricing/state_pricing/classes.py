import pandas as pd
import numpy as np


class StatePricing:

    def __init__(self, init_price_series: pd.Series, state_payoff_df: pd.DataFrame, physical_prob_series=None):
        price_s = init_price_series.copy()
        self.init_price_series = price_s

        payoff_df = state_payoff_df.copy()

        self.state_payoff_df = payoff_df

        self.physical_prob_series = physical_prob_series

    @property
    def physical_prob_series(self):
        if self._prob_s is None:
            raise AttributeError('physical_prob_series was not set!')
        return self._prob_s

    @physical_prob_series.setter
    def physical_prob_series(self, value: pd.Series):
        if value is not None:
            assert (self.state_payoff_df.columns == value.index).all()

        self._prob_s = value

    @property
    def state_price(self):
        state_price = self.init_price_series.values @ np.linalg.inv(self.state_payoff_df.values.T)
        state_price = pd.Series(state_price, index=self.state_payoff_df.columns, name='state price')
        return state_price

    def risk_neutral_vector(self, risk_free_factor):
        state_price = self.state_price
        risk_neutral = state_price * risk_free_factor
        risk_neutral = pd.Series(risk_neutral, index=state_price.index, name='risk neutral vector')
        return risk_neutral

    @property
    def pricing_kernel(self):
        pricing_kernel = self.state_price / self.physical_prob_series
        pricing_kernel.name = 'pricing kernel'
        return pricing_kernel

    def radon_nikodym_d(self, risk_free_factor):
        rn = risk_free_factor * self.pricing_kernel
        rn.name = 'radon nikodym derivative'
        return rn

    def _get_underlying(self):
        underlying_list = self.state_payoff_df.index.to_list()
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

    def payoff_valuation(self, payoff, risk_free_factor=1.0, risk_neutral=True):
        if risk_neutral:
            return (payoff * self.state_price).sum()
        else:
            return (payoff * self.physical_prob_series / risk_free_factor).sum()
