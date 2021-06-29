import numpy as np
from autograd import grad
import scipy.stats as st
from scipy import stats
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm
from scipy.special import logsumexp
from autograd.scipy import stats as autograd_stats


class Gaussian1D:
    def __init__(
        self, mean=None, variance=None, precision=None, precisionxmean=None
    ):
        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = 1.0 / variance
            self.precisionxmean = mean / variance
        else:
            self.precision = precision
            self.precisionxmean = precisionxmean
            self.variance = 1.0 / precision
            self.mean = precisionxmean / precision
        self.sigma = np.sqrt(self.variance)

    def sample(self, no_samples):
        return np.random.randn(no_samples) * np.sqrt(self.variance) + self.mean

    def logprob(self, x):
        return -(0.5 * (np.log(2 * np.pi * self.sigma * self.sigma) + ((x - self.mean) / self.sigma) ** 2))

class MVGaussian:
    def __init__(
        self, mean=None, variance=None, precision=None, precisionxmean=None
    ):
        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = np.linalg.inv(self.variance)
            self.precisionxmean = np.matmul(self.precision, self.mean)
        else:
            self.precision = precision
            self.precisionxmean = precisionxmean
            self.variance = np.linalg.inv(precision)
            self.mean = np.matmul(self.variance, precisionxmean)
        #self.sigma = np.sqrt(self.variance)

    def sample(self, no_samples):
        return np.random.multivariate_normal(self.mean, self.variance, size=(no_samples,))
        #return np.random.randn(no_samples) * np.sqrt(self.variance) + self.mean

    def logprob(self, x):
        y = multivariate_normal.pdf(x, mean=self.mean, cov=self.variance)
        #return -(0.5 * (np.log(2 * np.pi * self.sigma * self.sigma) + ((x - self.mean) / self.sigma) ** 2))


class Student1D:
    def __init__(
        self, q=None, mean=None, variance=None, precision=None, precisionxmean=None, df = None
    ):
        self.df = (3-q)/(q-1) if (q is not None and q!=1) else df
        #self.q = q if q is not None else
        if self.df is None:
            raise ValueError("Please specify q parameter (q!=1) or df (degrees of freedom)")

        self.q = q if q is not None else (self.df + 3)/(self.df + 1)

        if mean is not None:
            self.mean = mean
            self.variance = variance
            self.precision = 1.0 / variance
            self.precisionxmean = mean / variance
        else:
            if precisionxmean is not None:
                self.precision = precision
                self.precisionxmean = precisionxmean
                self.variance = 1.0 / precision
                self.mean = precisionxmean / precision
            else:
                self.mean = 0
                self.variance = 1
                self.precision = 1.0 / variance
                self.precisionxmean = mean / variance
        self.sigma = np.sqrt(self.variance)

        # RV with functions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
        #    e.g. logpdf(x, df, loc=0, scale=1),
        #         rvs(df, loc=0, scale=1, size=1, random_state=None) (sampling)
        self.student_t = stats.t
        self.student_t_autograd = autograd_stats.t

    def sample(self, no_samples):
        return self.student_t.rvs(self.df, size=no_samples, loc = self.mean, scale = self.sigma)
    def logprob(self, x):
        return self.student_t_autograd.logpdf(x, self.df, loc = self.mean, scale = self.sigma)




# class MVStudent: # TO DO:
#     def __init__(
#         self, q=None, mean=None, variance=None, precision=None, precisionxmean=None, df = None
#     ):

#         d = mean.shape[0] if mean is not None else precision.shape[0]
#         self.df = (d-d*q+2)/(q-1) if (q is not None and q!=1) else df
#         #self.df = (3-q)/(q-1) if (q is not None and q!=1) else df
#         if self.df is None:
#             raise ValueError("Please specify q parameter (q!=1) or df (degrees of freedom)")

#         self.q = q if q is not None else (self.df + d + 2)/(self.df + d)

#         if mean is not None:
#             self.mean = mean
#             self.variance = variance
#             self.precision = np.linalg.inv(self.variance)
#             self.precisionxmean = np.matmul(self.precision, self.mean)
#         else:
#             self.precision = precision
#             self.precisionxmean = precisionxmean
#             self.variance = np.linalg.inv(precision)
#             self.mean = np.matmul(self.variance, precisionxmean)

#         self.mvt = stats.multivariate_t(loc = self.mean, shape = self.variance, df = self.df, allow_singular=False)
#         # self.sigma = np.sqrt(self.variance)

#         # RV with functions: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
#         #    e.g. logpdf(x, df, loc=0, scale=1),
#         #         rvs(df, loc=0, scale=1, size=1, random_state=None) (sampling)


#     def sample(self, no_samples):
#         return self.mvt.rvs(size=(no_samples,))
#         #return self.student_t.rvs(self.df, size=no_samples, loc = self.mean, scale = self.sigma)
#         #np.random.randn(no_samples) * np.sqrt(self.variance) + self.mean

#     def logprob(self, x):
#         return self.mvt.logpdf(x, loc = self.mean, scale = self.sigma, df = self.df)


def geometric_average(dist1, dist2, pow1=0.5, pow2=0.5):
    precision = dist1.precision * pow1 + dist2.precision * pow2
    precisionxmean = dist1.precisionxmean * pow1 + dist2.precisionxmean * pow2

    if isinstance(dist1.precision, float) or isinstance(dist1.precision, int):
        return Gaussian1D(precision=precision, precisionxmean=precisionxmean)
    else:
        return MVGaussian(precision=precision, precisionxmean=precisionxmean)



def moment_average(dist1, dist2, beta):
    pow1 = 1 - beta
    pow2 = beta

    mean = pow1 * dist1.mean + pow2 * dist2.mean
    var = (
        pow1 * dist1.variance
        + pow2 * dist2.variance
        + pow1 * pow2 * np.outer(dist1.mean - dist2.mean, dist1.mean - dist2.mean)
    )
    var = np.abs(var) if len(mean.shape)==1 else var # TODO: fix small negative variance, check invertibility for >1d

    if isinstance(dist1.mean, float) or isinstance(dist1.mean, int):
        return Gaussian1D(mean=mean, variance=var)
    else:
        return MVGaussian(mean=mean, variance=var)


def student_moment_average(dist1, dist2, beta):
    pow1 = 1 - beta
    pow2 = beta

    # See Sec. 5.2 of Overleaf (esp. Eq 37-38)
    t_scaling = (dist2.df+2) / dist2.df

    t_mean = pow1 * dist1.mean + pow2 * dist2.mean
    t_var = ( pow1 * dist1.variance
             + pow2 * dist2.variance
             + t_scaling * pow1 * pow2 * np.outer(dist2.mean - dist1.mean, dist2.mean - dist1.mean)     )


    return Student1D(df = dist2.df, mean=t_mean, variance=t_var)
    # if isinstance(dist1.mean, float) or isinstance(dist1.mean, int):
    #     return Student1D(df = dist2.df, mean=t_mean, variance=t_var)
    # else:
    #     return MVStudent(mean = t_mean, variance = t_var, df = dist2.df)



def run_alpha_average_loops(dist1, dist2, beta, alpha, no_iters=500):
    ''' works for exp family endpoints only (as is, at least):  e.g. geometric mixture of Student-Ts is not Student-T or Gaussian'''
    q = deepcopy(dist1)
    for i in range(no_iters):
        dist2hat = geometric_average(dist2, q, alpha, 1.0 - alpha)
        dist1hat = geometric_average(dist1, q, alpha, 1.0 - alpha)
        q = moment_average(dist1hat, dist2hat, 1 - beta, beta)
    return q


def find_average_batch(dist1, dist2, beta_list, option):
    dists = []
    for beta in beta_list:
        if option == "moment":
            dists.append(moment_average(dist1, dist2, 1.0 - beta, beta))
        elif option == "geometric":
            dists.append(geometric_average(dist1, dist2, 1.0 - beta, beta))
        else:
            raise NotImplementedError("unknown option")
    return dists


def find_alpha_average_batch(
    dist1, dist2, beta_list, alpha_list, no_iters=500
):
    aa_dists = []
    for alpha in alpha_list:
        aa_alpha = []
        for beta in beta_list:
            aa_alpha.append(
                run_alpha_average_loops(dist1, dist2, beta, alpha, no_iters)
            )
        aa_dists.append(aa_alpha)
    return aa_dists


def run_ais(beta_dists, no_samples):
    no_betas = len(beta_dists)
    dist0 = beta_dists[0]
    x = dist0.sample(no_samples)
    logw = np.zeros(no_samples)
    logwt = np.zeros(no_samples)
    for i in range(1, no_betas):
        di = beta_dists[i]
        dim1 = beta_dists[i - 1]
        # update lower bound weights
        logw = logw + di.logprob(x) - dim1.logprob(x)
        # perfect transition
        x = di.sample(no_samples)
        # update upper bound weights
        logwt = logwt + di.logprob(x) - dim1.logprob(x)

    logZ = logsumexp(logw) - np.log(no_samples)
    logZ_lower = np.mean(logw)
    logZ_upper = np.mean(logwt)
    return logZ, logZ_lower, logZ_upper


def find_ti_integrand_geometric(dists):
    mean1, var1 = dists[0].mean, dists[0].variance
    mean2, var2 = dists[-1].mean, dists[-1].variance
    mean_list = np.array([dist.mean for dist in dists])
    var_list = np.array([dist.variance for dist in dists])
    const_term = -0.5 * np.log(var2 / var1)
    f1 = 0.5 / var1 * ((mean_list - mean1) ** 2 + var_list)
    f2 = -0.5 / var2 * ((mean_list - mean2) ** 2 + var_list)
    f = f1 + f2 + const_term
    return f


def find_sum_elements_geometric(dists, beta_vec):
    f = find_ti_integrand_geometric(dists)
    elements_lower = f[:-1] * (beta_vec[1:] - beta_vec[:-1])
    elements_upper = f[1:] * (beta_vec[1:] - beta_vec[:-1])
    return elements_lower, elements_upper


def find_sum_elements_moment(dists, beta_vec):
    mean1, var1 = dists[0].mean, dists[0].variance
    mean2, var2 = dists[-1].mean, dists[-1].variance
    mean_list = np.array([dist.mean for dist in dists])
    var_list = np.array([dist.variance for dist in dists])
    logCb = (
        (1 - beta_vec) * -0.5 * np.log(2 * np.pi * var1)
        + beta_vec * -0.5 * np.log(2 * np.pi * var2)
        - 0.5 * mean1 ** 2 * (1 - beta_vec) / var1
        - 0.5 * mean2 ** 2 * beta_vec / var2
        + 0.5 * mean_list ** 2 / var_list
    )

    mk, mkm1 = mean_list[1:], mean_list[:-1]
    vk, vkm1 = var_list[1:], var_list[:-1]
    f1 = -0.5 / vk * ((mk - mkm1) ** 2 + vkm1 - vk)
    logCk, logCkm1 = logCb[1:], logCb[:-1]
    f2 = logCk - logCkm1
    elements_lower = f1 + f2

    f1 = 0.5 / vkm1 * ((mk - mkm1) ** 2 + vk - vkm1)
    logCk, logCkm1 = logCb[1:], logCb[:-1]
    f2 = logCk - logCkm1
    elements_upper = f1 + f2

    return elements_lower, elements_upper
