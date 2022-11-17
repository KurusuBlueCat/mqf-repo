from mqf_toolbox.stochastic_modelling.option_pricer import *
# unit tests via grid of strike prices and T to maturity with different r for each T


S_test = 3662.45
K_test = np.linspace(3000, 4000, 20)
r_test = np.linspace(0.01, 0.14, 17)[:, np.newaxis] / 100.0
T_test = np.linspace(1, 17, 17)[:, np.newaxis] / 365
F_test = (S_test * np.exp(r_test * T_test))


def test_baseline_model():
    # Sanity Check with Prof Tee's function
    bs = BlackScholes(S=S_test, r=r_test, T=T_test)
    assert np.isclose(bs.vanilla_call(K=K_test, sigma=0.1),
                      BlackScholesCall(S=S_test, K=K_test, r=r_test, sigma=0.1, T=T_test)).all()


def test_black_scholes():
    # Put Call Parity Unit Test
    bs = BlackScholes(S=S_test, r=r_test, T=T_test)
    assert np.isclose(bs.vanilla_call(K=K_test, sigma=0.1) - bs.vanilla_put(K=K_test, sigma=0.1),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (bs.vanilla_call(K=K_test, sigma=0.1) >= 0).all()
    assert (bs.vanilla_put(K=K_test, sigma=0.1) >= 0).all()


def test_black_76():
    bs76 = Black76(F=F_test, r=r_test, T=T_test)
    assert np.isclose(bs76.vanilla_call(K=K_test, sigma=0.1) - bs76.vanilla_put(K=K_test, sigma=0.1),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (bs76.vanilla_call(K=K_test, sigma=0.1) >= 0).all()
    assert (bs76.vanilla_put(K=K_test, sigma=0.1) >= 0).all()


def test_bachelier():
    ba = Bachelier(F=F_test, r=r_test, T=T_test)
    assert np.isclose(ba.vanilla_call(K=K_test, sigma=0.1) - ba.vanilla_put(K=K_test, sigma=0.1),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (ba.vanilla_call(K=K_test, sigma=0.1) >= 0).all()
    assert (ba.vanilla_put(K=K_test, sigma=0.1) >= 0).all()


def test_dd():
    dd7 = DisplaceDiffusion(F=F_test, r=r_test, T=T_test, beta=0.7)
    assert np.isclose(dd7.vanilla_call(K=K_test, sigma=0.1) - dd7.vanilla_put(K=K_test, sigma=0.1),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (dd7.vanilla_call(K=K_test, sigma=0.1) >= 0).all()
    assert (dd7.vanilla_put(K=K_test, sigma=0.1) >= 0).all()

    dd3 = DisplaceDiffusion(F=F_test, r=r_test, T=T_test, beta=0.3)
    assert np.isclose(dd3.vanilla_call(K=K_test, sigma=0.1) - dd3.vanilla_put(K=K_test, sigma=0.1),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (dd3.vanilla_call(K=K_test, sigma=0.1) >= 0).all()
    assert (dd3.vanilla_put(K=K_test, sigma=0.1) >= 0).all()


def test_sabr():
    alpha_test = 1.81727308
    beta_test = 0.7
    rho_test = -0.40460926
    nu_test = 2.78934577

    sabr = SABR(F=F_test, r=r_test, T=T_test, alpha=alpha_test, beta=beta_test, rho=rho_test, nu=nu_test)
    assert np.isclose(sabr.vanilla_call(K=K_test) - sabr.vanilla_put(K=K_test),
                      S_test - K_test * np.exp(-r_test * T_test)).all()

    assert (sabr.vanilla_call(K=K_test) >= 0).all()
    assert (sabr.vanilla_put(K=K_test) >= 0).all()


def test_dd_limits():
    dd = DisplaceDiffusion(F=F_test, r=r_test, T=T_test)
    bs = BlackScholes(S=S_test, r=r_test, T=T_test)
    ba = Bachelier(F=F_test, r=r_test, T=T_test)
    # DisplaceDiffusion beta = 1 should be equal to black scholes
    assert np.isclose(dd.vanilla_call(K=K_test, sigma=0.5, beta=1), bs.vanilla_call(K=K_test, sigma=0.5)).all()
    assert np.isclose(dd.vanilla_put(K=K_test, sigma=0.5, beta=1), bs.vanilla_put(K=K_test, sigma=0.5)).all()
    # DisplaceDiffusion beta near 0 should be close to bachelier
    assert np.isclose(dd.vanilla_call(K=K_test, sigma=0.5, beta=0.0000001), ba.vanilla_call(K=K_test, sigma=0.5)).all()
    assert np.isclose(dd.vanilla_put(K=K_test, sigma=0.5, beta=0.0000001), ba.vanilla_put(K=K_test, sigma=0.5)).all()
