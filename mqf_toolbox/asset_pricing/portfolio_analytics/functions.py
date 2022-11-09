import numpy as np


def get_greeks(returns_array, return_inv_r=False) -> tuple:
    """
    Get greeks from arrays of returns
    Use for plotting of efficient frontier

    Parameters
    ----------
    returns_array: [pd.DataFrame, np.array]
        mxn matrix/array/dataframe
        where n is the number of assets
        and m is the sample size
        returns are in percentage unit (3 is 3%)
    return_inv_r: bool = False
        if True, will also return inverse of covariance and expected_returns

    Returns
    -------
    alpha: float
        R' x V_-1 x e
    zeta: float
        R' x V_-1 x R
    delta: float
        e' x V_-1 x e
    inv_cov_m: np.matrix
        inverse of covariance of returns_array, shape nxn
    r: np.matrix
        expected returns of returns_array, shape nx1
    """

    cov = np.matrix(np.cov(returns_array.T))
    n = cov.shape[0]

    inv_cov_m = np.linalg.inv(cov)
    e_v = np.matrix(np.ones((n, 1)))
    r = np.matrix(np.mean(returns_array, axis=0)).reshape((-1, 1))

    alpha = r.T * inv_cov_m * e_v  # these are all np.matrix, so it is automatically calling matmul
    zeta = r.T * inv_cov_m * r
    delta = e_v.T * inv_cov_m * e_v

    if return_inv_r:
        return alpha.A[0][0], zeta.A[0][0], delta.A[0][0], inv_cov_m, r

    return alpha.A[0][0], zeta.A[0][0], delta.A[0][0]


def greeks_to_param(alpha, zeta, delta):
    """
    a simple transformation from the efficient frontier greeks
    to parameter for a y = c_1 + c_2(x - rm_v)**2 function

    Parameters
    ----------
    alpha: float
        R' x V_-1 x e
    zeta: float
        R' x V_-1 x R
    delta: float
        e' x V_-1 x e

    Returns
    -------
    mv: float
        minimum variance
    c: float
        coefficient of x
    min_variance_returns: float
        return of minimum variance portfolio

    """
    r_mv = alpha / delta
    mv = 1 / delta
    c = delta / (zeta * delta - alpha ** 2)

    return mv, c, r_mv


def get_efficient_frontier_function(mv, c, r_mv, sqrt=False):
    """
    Converts mv, c, min_variance_returns into
    variance = mv + c(x - min_variance_returns)**2 function

    Parameters
    ----------
    mv: float
        minimum variance
    c: float
        coefficient of x
    r_mv: float
        returns of minimum variance portfolio
    sqrt: bool
        set output of the function to variance if false, standard deviation if true

    Returns
    -------
    eff: function(x: [float, np.array])
        this function computes mv + c(x - min_variance_returns)**2 for a given x
    """

    def eff(x):
        var = mv + c * (x - r_mv) ** 2
        return var ** 0.5 if sqrt else var

    return eff
