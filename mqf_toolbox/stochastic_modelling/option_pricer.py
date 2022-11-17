import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from scipy.optimize import brentq
from ..utils import MustSet


class BasePricer(metaclass=ABCMeta):
    """
    Base class for easy initialization of pricing calculation
    """
    S = MustSet()
    K = MustSet()
    r = MustSet()
    sigma = MustSet()
    T = MustSet()

    def __repr__(self):
        out_str = str(self.__class__.__base__.__name__)
        out_str += ': ' + str(self.__class__.__name__) + '\n\t'
        return out_str + self._params_status_str(['S', 'K', 'r', 'sigma', 'T'])

    def __init__(self, S=None, K=None, r=None, sigma=None, T=None, verbose=False):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.verbose = verbose

    def _get_params(self, **kwargs):
        attr = [getattr(self, keys) if value is None else value for keys, value in kwargs.items()]
        return attr

    def _get_params_status(self, arg_name_list):
        status_dict = {}
        for arg_name in arg_name_list:
            try:
                status_dict[arg_name] = getattr(self, arg_name)
            except AttributeError:
                status_dict[arg_name] = 'Not Set'

        return status_dict

    def _params_status_str(self, arg_name_list):
        status_dict = self._get_params_status(arg_name_list)
        out_str = ''
        for k, v in status_dict.items():
            out_str = out_str + str(k) + ': ' + str(v) + '\n\t'

        return out_str

    @abstractmethod
    def cash_nothing_call(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def cash_nothing_put(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def asset_nothing_call(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def asset_nothing_put(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    def vanilla_call(self, **kwargs):
        kwargs_list = self._get_params(**kwargs)
        kwargs = {k: value for k, value in zip(list(kwargs.keys()), kwargs_list)}
        K = self._get_params(K=kwargs.get('K', None))[0]

        asset = self.asset_nothing_call(**kwargs)
        strike = K * self.cash_nothing_call(**kwargs)

        if self.verbose:
            print(kwargs)
            print('Asset or Nothing call valued at: ' + str(asset))
            print('Cash or Nothing call valued at: ' + str(strike))

        return asset - strike

    def vanilla_put(self, **kwargs):
        kwargs_list = self._get_params(**kwargs)
        kwargs = {k: value for k, value in zip(list(kwargs.keys()), kwargs_list)}
        K = self._get_params(K=kwargs.get('K', None))[0]

        asset = self.asset_nothing_put(**kwargs)
        strike = K * self.cash_nothing_put(**kwargs)

        if self.verbose:
            print(kwargs)
            print('Asset or Nothing put valued at:' + str(asset))
            print('Cash or Nothing put valued at:' + str(strike))

        return strike - asset


class ForwardPricer(BasePricer):
    F = MustSet()

    def __init__(self, F=None, K=None, r=None, sigma=None, T=None, verbose=False):
        super().__init__(S=None, K=K, r=r, sigma=sigma, T=T, verbose=verbose)
        self.F = F

    def __repr__(self):
        out_str = str(self.__class__.__base__.__name__)
        out_str += ': ' + str(self.__class__.__name__) + '\n\t'
        return out_str + self._params_status_str(['F', 'K', 'r', 'sigma', 'T'])

    @abstractmethod
    def cash_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def cash_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def asset_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented

    @abstractmethod
    def asset_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)  # noqa
        return NotImplemented


# Model Implementation Starts here
class BlackScholes(BasePricer):
    """
    Good ol BlackScholes
    """
    def cash_nothing_call(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)
        d2 = np.log(S / K)
        d2 = d2 + (r - (sigma ** 2) / 2) * T
        d2 = d2 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def cash_nothing_put(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)
        d2 = np.log(K / S)
        d2 = d2 - (r - (sigma ** 2) / 2) * T
        d2 = d2 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def asset_nothing_call(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)
        d1 = np.log(S / K)
        d1 = d1 + (r + (sigma ** 2) / 2) * T
        d1 = d1 / (sigma * np.sqrt(T))

        return S * norm.cdf(d1)

    def asset_nothing_put(self, S=None, K=None, r=None, sigma=None, T=None):
        S, K, r, sigma, T = self._get_params(S=S, K=K, r=r, sigma=sigma, T=T)
        d1 = np.log(K / S)
        d1 = d1 - (r + (sigma ** 2) / 2) * T
        d1 = d1 / (sigma * np.sqrt(T))

        return S * norm.cdf(d1)

    def implied_vol(self, price, payoff, S=None, K=None, r=None, T=None):
        S, K, r, T = self._get_params(S=S, K=K, r=r, T=T)

        try:
            if payoff == 'call':
                impliedVol = brentq(lambda x: price - self.vanilla_call(S=S, K=K, r=r, sigma=x, T=T),
                                    1e-6, 50)

                return impliedVol

            elif payoff == 'put':
                impliedVol = brentq(lambda x: price - self.vanilla_put(S=S, K=K, r=r, sigma=x, T=T),
                                    1e-6, 50)

                return impliedVol
        except ValueError:
            return np.nan

        raise Exception("Payoff should be 'put' or 'call'")


class Black76(ForwardPricer):
    """
    BlackScholes with martingale
    """

    def cash_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d2 = np.log(F / K)
        d2 = d2 - ((sigma ** 2) / 2) * T  # no r_test, negative
        d2 = d2 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def cash_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d2 = np.log(K / F)
        d2 = d2 + ((sigma ** 2) / 2) * T  # no r_test, positive
        d2 = d2 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def asset_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d1 = np.log(F / K)
        d1 = d1 + ((sigma ** 2) / 2) * T  # no r_test
        d1 = d1 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * F * norm.cdf(d1)  # has discount

    def asset_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d1 = np.log(K / F)
        d1 = d1 - ((sigma ** 2) / 2) * T  # no r_test
        d1 = d1 / (sigma * np.sqrt(T))

        return np.exp(-r * T) * F * norm.cdf(d1)  # has discount

    def implied_vol(self, price, payoff, F=None, K=None, r=None, T=None):
        F, K, r, T = self._get_params(F=F, K=K, r=r, T=T)

        try:
            if payoff == 'call':
                impliedVol = brentq(lambda x: price - self.vanilla_call(F=F, K=K, r=r, sigma=x, T=T),
                                    1e-6, 50)

                return impliedVol

            elif payoff == 'put':
                impliedVol = brentq(lambda x: price - self.vanilla_put(F=F, K=K, r=r, sigma=x, T=T),
                                    1e-6, 50)

                return impliedVol
        except ValueError:
            return np.nan

        raise Exception("Payoff should be 'put' or 'call'")


class Bachelier(ForwardPricer):
    """
    Forward Bachelier
    """
    def cash_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d2 = F - K
        d2 = d2 / (sigma * F * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def cash_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d2 = K - F
        d2 = d2 / (sigma * F * np.sqrt(T))

        return np.exp(-r * T) * norm.cdf(d2)

    def asset_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d1 = F - K
        d1 = d1 / (sigma * F * np.sqrt(T))

        rT_S = F * norm.cdf(d1)
        sigma_T = sigma * F * np.sqrt(T) * norm.pdf(d1)

        return np.exp(-r * T) * (rT_S + sigma_T)

    def asset_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None):
        F, K, r, sigma, T = self._get_params(F=F, K=K, r=r, sigma=sigma, T=T)
        d1 = K - F
        d1 = d1 / (sigma * F * np.sqrt(T))

        rT_S = F * norm.cdf(d1)
        sigma_T = sigma * F * np.sqrt(T) * norm.pdf(d1)

        return np.exp(-r * T) * (rT_S - sigma_T)


class DisplaceDiffusion(Black76):
    """
    The super() call here will refer to the method implemented by Black76
    """
    beta = MustSet()

    def __repr__(self):
        out_str = super().__repr__()
        out_str += self._params_status_str(['beta'])
        return out_str

    def __init__(self, F=None, K=None, r=None, sigma=None, T=None, beta=None, verbose=False):
        super().__init__(F=F, K=K, r=r, sigma=sigma, T=T, verbose=verbose)
        self.beta = beta

    @staticmethod
    def _to_black76(F, K, r, sigma, T, beta):
        return F / beta, K + ((1 - beta) / beta) * F, r, sigma * beta, T

    @staticmethod
    def _displace(x, beta, a):
        return beta * x + (1 - beta) * a

    @classmethod
    def sigma_black_scholes(cls, F, K, sigma, T, A, beta):
        # Choi, J., Kwak, M., Tee, C. W., & Wang, Y. (2022).
        # A Black–Scholes user's guide to the Bachelier model.
        # Journal of Futures Markets, 42(5), 959-980.
        k = K / F
        k_d = cls._displace(K, beta, A) / cls._displace(F, beta, A)

        first_chunk = sigma * (cls._displace(F, beta, A) / F) * np.sqrt(k_d / k) * (
                    (1 + (np.log(k_d) ** 2) / 24) / (1 + (np.log(k) ** 2) / 24))
        second_chunk = 1 + (sigma ** 2) * ((cls._displace(F, beta, A) / F) ** 2) * (k_d / k) * (T / 24)
        third_chunk = 1 + (beta ** 2) * (sigma ** 2) * (T / 24)

        return first_chunk * second_chunk / third_chunk

    def cash_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None, beta=None):
        F, K, r, sigma, T = self._to_black76(*self._get_params(F=F, K=K, r=r, sigma=sigma, T=T, beta=beta))
        return super().cash_nothing_call(F, K, r, sigma, T)

    def cash_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None, beta=None):
        F, K, r, sigma, T = self._to_black76(*self._get_params(F=F, K=K, r=r, sigma=sigma, T=T, beta=beta))
        return super().cash_nothing_put(F, K, r, sigma, T)

    def asset_nothing_call(self, F=None, K=None, r=None, sigma=None, T=None, beta=None):
        dd_F, beta = self._get_params(F=F, beta=beta)
        F, K, r, sigma, T = self._to_black76(*self._get_params(F=F, K=K, r=r, sigma=sigma, T=T, beta=beta))

        built_in_cash = dd_F * ((1 / beta) - 1) * super().cash_nothing_call(F, K, r, sigma, T)
        asset = super().asset_nothing_call(F, K, r, sigma, T)

        return asset - built_in_cash

    def asset_nothing_put(self, F=None, K=None, r=None, sigma=None, T=None, beta=None):
        dd_F, beta = self._get_params(F=F, beta=beta)
        F, K, r, sigma, T = self._to_black76(*self._get_params(F=F, K=K, r=r, sigma=sigma, T=T, beta=beta))

        built_in_cash = dd_F * ((1 / beta) - 1) * super().cash_nothing_put(F, K, r, sigma, T)
        asset = super().asset_nothing_put(F, K, r, sigma, T)

        return asset - built_in_cash


class SABR(Black76):
    """
    SABR model implemented via calculation of blackscholes implied vol,
    then passed that vol back to black 76 model
    """
    alpha = MustSet()
    beta = MustSet()
    rho = MustSet()
    nu = MustSet()

    def __repr__(self):
        out_str = str(self.__class__.__base__.__name__)
        out_str += ': ' + str(self.__class__.__name__) + '\n\t'
        return out_str + self._params_status_str(['F',
                                                  'K',
                                                  'r',
                                                  'alpha',
                                                  'beta',
                                                  'rho',
                                                  'nu',
                                                  'T'])

    def __init__(self, F=None, K=None, r=None, alpha=None, beta=None, rho=None,
                 nu=None, T=None, verbose=False):
        super().__init__(F=F, K=K, r=r, sigma=None, T=T, verbose=verbose)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    @staticmethod
    def sigma_black_scholes(F, K, T, alpha, beta, rho, nu):
        # simplified to facilitate easy vectorization
        z = (nu / alpha) * ((F * K) ** (0.5 * (1 - beta))) * np.log(F / K)
        zhi = np.log((((1 - 2 * rho * z + z * z) ** 0.5) + z - rho) / (1 - rho))
        numer1 = (((1 - beta) ** 2) / 24) * ((alpha * alpha) / ((F * K) ** (1 - beta)))
        numer2 = 0.25 * rho * beta * nu * alpha / ((F * K) ** ((1 - beta) / 2))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
        denom1 = ((1 - beta) ** 2 / 24) * (np.log(F / K)) ** 2
        denom2 = (((1 - beta) ** 4) / 1920) * ((np.log(F / K)) ** 4)
        denom = ((F * K) ** ((1 - beta) / 2)) * (1 + denom1 + denom2) * zhi

        return numer / denom

    def cash_nothing_call(self, F=None, K=None, r=None, alpha=None, beta=None, rho=None,
                          nu=None, T=None):
        F, K, r, alpha, beta, rho, nu, T = self._get_params(F=F, K=K, r=r,
                                                            alpha=alpha, beta=beta, rho=rho,  # noqa
                                                            nu=nu, T=T)  # noqa

        sigma_bs = self.sigma_black_scholes(F, K, T, alpha, beta, rho, nu)
        return super().cash_nothing_call(F=F, K=K, r=r, sigma=sigma_bs, T=T)

    def cash_nothing_put(self, F=None, K=None, r=None, alpha=None, beta=None, rho=None,
                         nu=None, T=None):
        F, K, r, alpha, beta, rho, nu, T = self._get_params(F=F, K=K, r=r,
                                                            alpha=alpha, beta=beta, rho=rho,  # noqa
                                                            nu=nu, T=T)  # noqa

        sigma_bs = self.sigma_black_scholes(F, K, T, alpha, beta, rho, nu)
        return super().cash_nothing_put(F=F, K=K, r=r, sigma=sigma_bs, T=T)

    def asset_nothing_call(self, F=None, K=None, r=None, alpha=None, beta=None, rho=None,
                           nu=None, T=None):
        F, K, r, alpha, beta, rho, nu, T = self._get_params(F=F, K=K, r=r,
                                                            alpha=alpha, beta=beta, rho=rho,  # noqa
                                                            nu=nu, T=T)  # noqa

        sigma_bs = self.sigma_black_scholes(F, K, T, alpha, beta, rho, nu)
        return super().asset_nothing_call(F=F, K=K, r=r, sigma=sigma_bs, T=T)

    def asset_nothing_put(self, F=None, K=None, r=None, alpha=None, beta=None, rho=None,
                          nu=None, T=None):
        F, K, r, alpha, beta, rho, nu, T = self._get_params(F=F, K=K, r=r,
                                                            alpha=alpha, beta=beta, rho=rho,  # noqa
                                                            nu=nu, T=T)  # noqa

        sigma_bs = self.sigma_black_scholes(F, K, T, alpha, beta, rho, nu)
        return super().asset_nothing_put(F=F, K=K, r=r, sigma=sigma_bs, T=T)


def BlackScholesCall(S, K, r, sigma, T):  # noqa
    # from prof Tee
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
