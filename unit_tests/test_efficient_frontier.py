from mqf_toolbox.asset_pricing.portfolio_analytics.analysis import *
from mqf_toolbox.asset_pricing.portfolio_analytics.plots import *

data = pd.read_csv('Industry_Portfolios.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data = data.set_index('Date')

market_data = pd.read_csv('Market_Portfolio.csv')
market_data['Date'] = pd.to_datetime(market_data['Date'], format='%d/%m/%Y')
market_data = market_data.set_index('Date').iloc[:, 0]

risk_data = pd.read_csv('Risk_Factors.csv')
risk_data['Date'] = pd.to_datetime(risk_data['Date'], format='%d/%m/%Y')
risk_data = risk_data.set_index('Date')

# ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=0.13)
# plot_eff(ap_frame, sqrt=True, mode='te')
# plot_tangent(ap_frame, ap_frame.bm_tangency_returns, mode='te', orthogonal=True)

# ap_frame.calculate_portfolio_weights(ap_frame.min_variance_returns)
# ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=risk_data['Rf'].to_frame())
# plot_montecarlo_portfolio_apf(ap_frame, mode='te', inverse=True)
# plot_eff(ap_frame, sqrt=True, mode='te', ylim=(-1, 1))


def test_alpha_beta():
    ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=risk_data['Rf'].to_frame())
    a = calculate_alpha_beta_statsmodels(ap_frame, risk_free_rate=risk_data['Rf'].to_frame())
    b = calculate_alpha_beta(ap_frame, risk_free_rate=risk_data['Rf'].to_frame())

    # calculate_alpha_beta(ap_frame)

    assert np.isclose(a, b).all()


def test_sharpe():
    ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=risk_data['Rf'].to_frame())
    sharpe1 = ap_frame.calculate_sharpe()
    sharpe2 = calculate_sharpe(ap_frame)

    assert np.isclose(sharpe1, sharpe2).all()
    assert np.isclose(sharpe1.mean(), 0.15793196229339906)


def test_tangency_weights_rf():
    ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=0.13)
    tangency_weights = ap_frame.ef.calculate_portfolio_weights(ap_frame.rf_tangency_returns)
    assert np.isclose(tangency_weights.std(), 0.5263772975082441)

    tangency_weights2 = ap_frame.calculate_weights_rf(ap_frame.rf_tangency_returns)
    assert np.isclose(tangency_weights, tangency_weights2).all()


def test_multi_factor_beta():
    ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=risk_data['Rf'].to_frame())
    manual_result = calculate_alpha_beta_multifactor(ap_frame, risk_data=risk_data.iloc[:, 1:])

    ap_frame = APFrame(data,
                       benchmark_series=market_data,
                       risk_factor_df=risk_data.iloc[:, 2:],
                       risk_free_rate=risk_data['Rf'].to_frame())

    infer_result = calculate_alpha_beta_multifactor(ap_frame)

    assert np.isclose(infer_result, manual_result).all()

    infer_result2 = calculate_alpha_beta_multifactor(ap_frame, include_bm=True)
    assert np.isclose(infer_result2.mean().mean(), 0.2759845734637398)


def test_sortino():
    ap_frame = APFrame(data, benchmark_series=market_data, risk_free_rate=risk_data['Rf'].to_frame())
    sortino1 = ap_frame.calculate_sortino(0.13)
    sortino2 = calculate_sortino(ap_frame, target=0.13)
    assert np.isclose(sortino1, sortino2).all()


def test_metrics():
    ap_frame = APFrame(data, benchmark_series=market_data,
                       risk_factor_df=risk_data.iloc[:, 2:],
                       risk_free_rate=risk_data['Rf'].to_frame())
    metrics = calculate_5_metrics(ap_frame)
    assert np.isclose(metrics.mean().mean(), 0.301362598870715)
    plots_metrics(metrics)

