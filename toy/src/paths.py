from src.hmc import HMCDist, gaus_log_prob
from src.ais import Student1D
import autograd.numpy as np
import numpy as onp

def geometric_average(proposal, target, beta):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    def neg_logp(x):
        log_prob = proposal.logprob(x) * pow1 + target.logprob(x) * pow2
        return -log_prob

    return HMCDist(neg_logp)


def moment_average(proposal, target, beta):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    mean = pow1 * proposal.mean + pow2 * target.mean
    var = (
        pow1 * proposal.variance
        + pow2 * target.variance
        + pow1 * pow2 * (proposal.mean - target.mean) ** 2
    )

    # will handle when we move to MV
    # var = (
    #     pow1 * proposal.variance
    #     + pow2 * target.variance
    #     + pow1 * pow2 * np.outer(proposal.mean - target.mean, proposal.mean - target.mean)
    # )

    var = onp.abs(var)  # TODO: fix small negative variance
    sig = onp.sqrt(var)

    def neg_logp(x):
        return -gaus_log_prob(x, mean, sig)

    return HMCDist(neg_logp)


def alpha_average(proposal, target, beta, alpha):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    def neg_logp(x):
        if alpha == 1:
            log_prob = pow1*proposal.logprob(x) + pow2*target.logprob(x)
        else:
            log_prob = (2/(1-alpha))*(np.logaddexp(
                np.log(pow1) + ((1-alpha)/2)* proposal.logprob(x),
                np.log(pow2) + ((1-alpha)/2)* target.logprob(x)))
        return -log_prob

    return HMCDist(neg_logp)


def qpath(proposal, target, beta, q):
    assert 0 <= beta <= 1
    if beta == 0: return proposal
    pow1, pow2 = 1.0 - beta, beta

    def neg_logp(x):
        if q == 1:
            log_prob = pow1*proposal.logprob(x) + pow2*target.logprob(x)
        else:
            log_prob = (1/(1-q))*(np.logaddexp(
                np.log(pow1) + (1-q)* proposal.logprob(x),
                np.log(pow2) + (1-q)* target.logprob(x)))
        return -log_prob

    return HMCDist(neg_logp)

def q_moment(dist1, dist2, beta):
    pow1 = 1 - beta
    pow2 = beta

    # See Sec. 5.2 of Overleaf (esp. Eq 37-38)
    t_scaling = (dist2.df+2) / dist2.df

    t_mean = pow1 * dist1.mean + pow2 * dist2.mean
    t_var = ( pow1 * dist1.variance
             + pow2 * dist2.variance
             + t_scaling * pow1 * pow2 * np.outer(dist2.mean - dist1.mean, dist2.mean - dist1.mean)     )

    return Student1D(df = dist2.df, mean=t_mean, variance=t_var)

