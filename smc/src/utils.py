import numpy as np
from scipy import optimize
from scipy.special import factorial
from tqdm.std import tqdm

from ml_helpers import BestMeter

def log_sum_weighted_exp(val1, val2, weight1, weight2):
    val_max = np.where(val1 > val2, val1, val2)
    val1_exp = weight1 * np.exp(val1 - val_max)
    val2_exp = weight2 * np.exp(val2 - val_max)
    lse = val_max + np.log(val1_exp + val2_exp)
    return lse

def logpdf_to_qlogpdf(logpdf, q):
    return (np.exp((1 - q) * logpdf) - 1) / (1 - q)

def essl(lw):
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw: (N,) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = np.exp(lw - lw.max())
    return (w.sum())**2 / np.sum(w**2)


def qdiff(log_b_over_a, beta1, beta2, q1=1, q2=1):
    """computes logqpath(beta1, q1) - logqpath(beta2, q2)"""

    if q1 == 1:
        qpath1 = beta1 * log_b_over_a
    else:
        qpath1 = (1/(1 - q1)) * np.log(1 + beta1*(np.exp((1-q1)*log_b_over_a) - 1))

    if q2 == 1:
        qpath2 = beta2 * log_b_over_a
    else:
        qpath2 = (1/(1 - q2)) * np.log(1 + beta2*(np.exp((1-q2)*log_b_over_a) - 1))

    return qpath1 - qpath2


def lnq_from_ln(lnx, q):
    return ( 1 / (1-q) ) * ( np.exp((1-q) * lnx) - 1)


def delta_to_q(delta):
    # 18 is precision in log_10, set to geometric
    return 1.0 if delta > 18 else 1 - 10**(-delta)

def learn_delta(log_b_over_a, beta, ESSmin, fudge, restarts=500):
    # eq. 39
    init_val = np.log10(np.abs(log_b_over_a).max())

    def _helper(adaptive_q_init_val):
        # ess objective
        f = lambda e: (essl(qdiff(log_b_over_a, e[0], beta, delta_to_q(e[1]), 1)) - ESSmin)**2
        new_e = optimize.minimize(f,
                                x0=(1 - fudge, adaptive_q_init_val),
                                # coordinate descent on beta, q
                                method='Powell',
                                bounds=[(beta, 1 - fudge),
                                        (0, 18)])
        new_beta, new_delta = new_e.x[0], new_e.x[1]
        return new_e.fun, new_beta, new_delta

    if restarts == 0: return beta, init_val

    # random restarts
    # q_init ~ N(init_val, 0.1)
    meter = BestMeter(mode='min', verbose=True)

    for i in range(restarts):
        # truncated normal around adaptive_q_init_val
        adaptive_q = init_val if i == 0 else np.clip(np.random.normal(init_val, 0.1), 0, 18)
        fun, beta, delta = _helper(adaptive_q)
        meter.step(fun, beta=beta, delta=delta)

    return meter.best_obj['beta'], meter.best_obj['delta']

