import numpy as np
import pandas as pd

from mqf_toolbox.asset_pricing.state_pricing import *


rf_factor = 1.1

prices = {
    'rf': 1,
    'A': 45,
    'B': 45}

state_phys_prob = {
    'Good': 0.3,
    'Normal': 0.5,
    'Bad': 0.2}

payoff = [[1.1, 1.1, 1.1],
          [75,  55,  20],
          [60,  50,  40], ]

state_df = pd.DataFrame(payoff, index=prices.keys(), columns=state_phys_prob.keys())

prices_s = pd.Series(prices, name='initial price')

phys_s = pd.Series(state_phys_prob, name='physical probability')

StatePricing(prices_s, state_df, physical_prob_series=phys_s).state_price


# s_pricing = StatePricing(prices_s, state_df, risk_free_rate=10, physical_prob_series=phys_s)
def test_state_price():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    assert np.isclose(s_pricing.state_price.values, np.array([0.22727273, 0.40909091, 0.27272727])).all()


def test_pricing_kernel():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    assert np.isclose(s_pricing.pricing_kernel, np.array([0.75757576, 0.81818182, 1.36363636])).all()


def test_radon_nikodym_derivatives():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    rn_radon = (s_pricing.state_payoff_df
                * s_pricing.physical_prob_series
                * s_pricing.radon_nikodym_d(rf_factor)
                / rf_factor).sum(1)

    rn_auto = (s_pricing.state_payoff_df * s_pricing.state_price).sum(1)

    assert np.isclose(rn_auto, rn_radon).all()


def test_call_payoff():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    all_payoff = s_pricing.calculate_call_payoff(100, ['A', 'B'])
    a_payoff = s_pricing.calculate_call_payoff(50, ['A'])

    assert np.isclose(all_payoff.values, np.array([35, 5, 0])).all()
    assert np.isclose(a_payoff.values, np.array([25, 5, 0])).all()


def test_put_payoff():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    all_payoff = s_pricing.calculate_put_payoff(100, ['A', 'B'])
    b_payoff = s_pricing.calculate_put_payoff(50, ['B'])

    assert np.isclose(all_payoff.values, np.array([0, 0, 40])).all()
    assert np.isclose(b_payoff.values, np.array([0, 0, 10])).all()


def test_valuation():
    s_pricing = StatePricing(prices_s, state_df, physical_prob_series=phys_s)
    put_payoff = s_pricing.calculate_put_payoff(100, ['A', 'B'])
    put_value_rn = s_pricing.payoff_valuation(put_payoff, risk_neutral=True)
    put_value = s_pricing.payoff_valuation(put_payoff, risk_free_factor=rf_factor, risk_neutral=False)

    assert np.isclose(put_value_rn, 10.9090909090909)
    assert np.isclose(put_value, 7.2727272727272725)
