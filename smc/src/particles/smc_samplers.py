# -*- coding: utf-8 -*-

"""
SMC samplers.

Overview
========

This module implements:

    * `StaticModel`: a base class for defining static models (and the
      corresponding target distributions).
    * `FeynmanKac` sub-classes that correspond to the following SMC samplers:
        + `IBIS`
        + `AdaptiveTempering` (and `Tempering` for the non-adaptive version)
        + `SMC2`
    * `ThetaParticles` classes: classes to represent a collection of N
      particles, together with extra fields, such as the posterior log-density.
      The particle systems generated by SMC samplers are objects of these classes.

For more information and examples, see the following notebook tutorial_.

.. _tutorial: notebooks/SMC_samplers_tutorial.ipynb

"""

from __future__ import absolute_import, division, print_function


import copy as cp
from src.qhelpers import delta_to_q, learn_delta, qdiff, qpath
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, LinAlgError, solve_triangular

from src import particles
from src.particles import resampling as rs
from src.particles.state_space_models import Bootstrap


###################################
# Static models


class StaticModel(object):
    """Base class for static models.

    To define a static model, sub-class `StaticModel`, and define method
    `logpyt`.

    Example
    -------
    ::

        class ToyModel(StaticModel):
            def logpyt(self, theta, t):
                return -0.5 * (theta['mu'] - self.data[t])**2

        my_toy_model = ToyModel(data=x, prior=pi)

    See doc of `__init__` for more details on the arguments
    """

    def __init__(self, data=None, prior=None):
        """
        Parameters
        ----------
        data: list-like
            data
        prior: `StructDist` object
            prior distribution of the parameters
        """
        self.data = data
        self.prior = prior

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def logpyt(self, theta, t):
        """log-likelihood of Y_t, given parameter and previous datapoints.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time
        """
        raise NotImplementedError('StaticModel: logpyt not implemented')

    def loglik(self, theta, t=None):
        """ log-likelihood at given parameter values.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time (if set to None, the full log-likelihood is returned)

        Returns
        -------
        l: float numpy.ndarray
            the N log-likelihood values
        """


        if t is None:
            t = self.T - 1
        l = np.zeros(shape=theta.shape[0])
        for s in range(t + 1):
            l += self.logpyt(theta, s)
        return l

    def logpost(self, theta, t=None):
        """Posterior log-density at given parameter values.

        Parameters
        ----------
        theta: dict-like
            theta['par'] is a ndarray containing the N values for parameter par
        t: int
            time (if set to None, the full posterior is returned)

        Returns
        -------
        l: float numpy.ndarray
            the N log-likelihood values
        """
        return self.prior.logpdf(theta) + self.loglik(theta, t)

###############################
# Theta Particles


def all_distinct(l, idx):
    """
    Returns the list [l[i] for i in idx] 
    When needed, objects l[i] are replaced by a copy, to make sure that
    the elements of the list are all distinct

    Parameters
    ---------
    l: iterable
    idx: iterable that generates ints (e.g. ndarray of ints)

    Returns
    -------
    a list
    """
    out = []
    deja_vu = [False for _ in l]
    for i in idx:
        to_add = cp.deepcopy(l[i]) if deja_vu[i] else l[i]
        out.append(to_add)
        deja_vu[i] = True
    return out


class FancyList(object):

    def __init__(self, l):
        self.l = l

    def __iter__(self):
        return iter(self.l)

    def __getitem__(self, key):
        if isinstance(key, np.ndarray):
            return FancyList(all_distinct(self.l, key))
        else:
            return self.l[key]

    def __setitem__(self, key, value):
        self.l[key] = value

    def copy(self):
        return cp.deepcopy(self)

    def copyto(self, src, where=None):
        """
        Same syntax and functionality as numpy.copyto

        """
        for n, _ in enumerate(self.l):
            if where[n]:
                self.l[n] = src.l[n]  # not a copy


def as_2d_array(theta):
    """ returns a view to record array theta which behaves
    like a (N,d) float array
    """
    v = theta.view(np.float)
    N = theta.shape[0]
    v.shape = (N, - 1)
    # raise an error if v cannot be reshaped without creating a copy
    return v


class ThetaParticles(object):
    """Base class for particle systems for SMC samplers.

    This is a rather generic class for packing together information on N
    particles; it may have the following attributes:

    * `theta`: a structured array (an array with named variables);
      see `distributions` module for more details on structured arrays.
    * a bunch of `numpy` arrays such that shape[0] = N; for instance an array
      ``lpost`` for storing the log posterior density of the N particles;
    * lists of length N; object n in the list is associated to particle n;
      for instance a list of particle filters in SMC^2; the name of each
      of of these lists must be put in class attribute *Nlists*.
    * a common attribute (shared among all particles).

    The whole point of this class is to mimic the behaviour of a numpy array
    containing N particles. In particular this class implements fancy
    indexing::

        obj[array([3, 5, 10, 10])]
        # returns a new instance that contains particles 3, 5 and 10 (twice)

    """
    shared = []  # put here the name of shared attributes

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.containers = [k for k in kwargs if k not in self.shared]
        if 'theta' in kwargs:
            self.arr = as_2d_array(self.theta)
            self.N, self.dim = self.arr.shape

    def __getitem__(self, key):
        attrs = {k: self.__dict__[k][key] for k in self.containers}
        if isinstance(key, int):
            return attrs
        else:
            attrs.update({k: cp.deepcopy(self.__dict__[k])
                          for k in self.shared})
            return self.__class__(**attrs)

    def __setitem__(self, key, value):
        for k in self.containers:
            self.__dict__[k][key] = value.__dict__[k]

    def copy(self):
        """Returns a copy of the object."""
        attrs = {k: self.__dict__[k].copy() for k in self.containers}
        attrs.update({k: cp.deepcopy(self.__dict__[k]) for k in self.shared})
        return self.__class__(**attrs)

    def copyto(self, src, where=None):
        """Emulates function `copyto` in NumPy.

       Parameters
       ----------

       where: (N,) bool ndarray
            True if particle n in src must be copied.
       src: (N,) `ThetaParticles` object
            source

       for each n such that where[n] is True, copy particle n in src
       into self (at location n)
        """
        for k in self.containers:
            v = self.__dict__[k]
            if isinstance(v, np.ndarray):
                np.copyto(v, src.__dict__[k], where=where)
            else:
                v.copyto(src.__dict__[k], where=where)

    def copyto_at(self, n, src, m):
        """Copy to at a given location.

        Parameters
        ----------
        n: int
            index where to copy
        src: `ThetaParticles` object
            source
        m: int
            index of the element to be copied

        Note
        ----
        Basically, does self[n] <- src[m]
        """
        for k in self.containers:
            self.__dict__[k][n] = src.__dict__[k][m]


class MetroParticles(ThetaParticles):
    """Particles that may be moved through a Metropolis step.

    The following attributes are required:
        * `theta`: a (N,) record array; the parameter values
        * `lpost`: a (N,) float array; log-posterior density at the parameter
          values

    An instance has the following shared attribute:
        * acc_rates: list; acceptance rates of the previous Metropolis steps

    This class implements generic methods to move all the particle
    according to a Metropolis step.
    """

    shared = ['acc_rates']

    def __init__(self, theta=None, lpost=None, acc_rates=None, **extra_kwargs):
        ThetaParticles.__init__(self, theta=theta, lpost=lpost,
                                **extra_kwargs)
        self.acc_rates = [] if acc_rates is None else acc_rates

    def mcmc_iterate(self, nsteps, xstart, xend, delta_dist):
        if nsteps == 0:
            prev_dist = 0.
            yield
            while True:
                mean_dist = np.mean(np.sqrt(np.sum((xend - xstart)**2, axis=1)))
                if np.abs(mean_dist - prev_dist) < delta_dist * prev_dist:
                    break
                prev_dist = mean_dist
                yield
        else:
            for _ in range(nsteps):
                yield

    class RandomWalkProposal(object):

        def __init__(self, x, scale=None, adaptive=True):
            if adaptive:
                if scale is None:
                    scale = 2.38 / np.sqrt(x.shape[1])
                cov = np.cov(x.T)
                try:
                    self.L = scale * cholesky(cov, lower=True)
                except LinAlgError:
                    self.L = scale * np.diag(np.sqrt(np.diag(cov)))
                    print('Warning: could not compute Cholesky decomposition, using diag matrix instead')
            else:
                if scale is None:
                    scale = 1.
                self.L = scale * np.eye(x.shape[1])

        def step(self, x):
            y = x + np.dot(stats.norm.rvs(size=x.shape), self.L.T)
            return y, 0.

    class IndependentProposal(object):

        def __init__(self, x, scale=1.1):
            self.L = scale * cholesky(np.cov(x.T), lower=True)
            self.mu = np.mean(x, axis=0)

        def step(self, x):
            z = stats.norm.rvs(size=x.shape)
            y = self.mu + np.dot(z, self.L.T)
            zx = solve_triangular(self.L, np.transpose(x - self.mu),
                                  lower=True)
            delta_lp = (0.5 * np.sum(z * z, axis=1)
                        - 0.5 * np.sum(zx * zx, axis=0))
            return y, delta_lp

    def choose_proposal(self, type_prop='random walk', rw_scale=None,
                        adaptive=True, indep_scale=1.1):
        if type_prop == 'random walk':
            return MetroParticles.RandomWalkProposal(self.arr,
                                                     scale=rw_scale,
                                                     adaptive=adaptive)
        if type_prop == 'independent':
            return MetroParticles.IndependentProposal(self.arr,
                                                      scale=indep_scale)
        raise ValueError('Unknown type for Metropolis proposal')

    def Metropolis(self, compute_target, mh_options):
        """Performs a certain number of Metropolis steps.

        Parameters
        ----------
        compute_target: function
            computes the target density for the proposed values
        mh_options: dict
            + 'type_prop': {'random walk', 'independent'}
              type of proposal: either Gaussian random walk, or independent Gaussian
            + 'adaptive': bool
              If True, the covariance matrix of the random walk proposal is
              set to a `rw_scale` times the weighted cov matrix of the particle
              sample (ignored if proposal is independent)
            + 'rw_scale': float (default=None)
              see above (ignored if proposal is independent)
            + 'indep_scale': float (default=1.1)
              for an independent proposal, the proposal distribution is
              Gaussian with mean set to the particle mean, cov set to
              `indep_scale` times particle covariance
            + 'nsteps': int (default: 0)
              number of steps; if 0, the number of steps is chosen adaptively
              as follows: we stop when the average distance between the
              starting points and the stopping points increase less than a
              certain fraction
            + 'delta_dist': float (default: 0.1)
              threshold for when nsteps = 0
        """
        opts = mh_options.copy()
        nsteps = opts.pop('nsteps', 0)
        delta_dist = opts.pop('delta_dist', 0.1)
        proposal = self.choose_proposal(**opts)
        xout = self.copy()
        xp = self.__class__(theta=np.empty_like(self.theta))
        step_ars = []
        for _ in self.mcmc_iterate(nsteps, self.arr, xout.arr, delta_dist):
            xp.arr[:, :], delta_lp = proposal.step(xout.arr)
            compute_target(xp)
            lp_acc = xp.lpost - xout.lpost + delta_lp
            accept = (np.log(stats.uniform.rvs(size=self.N)) < lp_acc)
            xout.copyto(xp, where=accept)
            step_ars.append(np.mean(accept))
        xout.acc_rates = self.acc_rates + [step_ars]
        return xout

#############################
# Basic importance sampler

class ImportanceSampler(object):
    """Importance sampler.

    Basic implementation of importance sampling, with the same interface
    as SMC samplers.

    Parameters
    ----------
    model: `StaticModel` object
        The static model that defines the target posterior distribution(s)
    proposal: `StructDist` object
        the proposal distribution (if None, proposal is set to the prior)

    """
    def __init__(self, model=None, proposal=None):
        self.proposal = model.prior if proposal is None else proposal
        self.model = model

    def run(self, N=100):
        """

        Parameter
        ---------
        N: int
            number of particles

        Returns
        -------
        wgts: Weights object
            The importance weights (with attributes lw, W, and ESS)
        X: ThetaParticles object
            The N particles (with attributes theta, logpost)
        norm_cst: float
            Estimate of the normalising constant of the target
        """
        th = self.proposal.rvs(size=N)
        self.X = ThetaParticles(theta=th, lpost=None)
        self.X.lpost = self.model.logpost(th)
        lw = self.X.lpost - self.proposal.logpdf(th)
        self.wgts = rs.Weights(lw=lw)
        self.norm_cst = rs.log_mean_exp(lw)

#############################
# FK classes for SMC samplers


class FKSMCsampler(particles.FeynmanKac):
    """Base FeynmanKac class for SMC samplers.

    Parameters
    ----------
    model: `StaticModel` object
        The static model that defines the target posterior distribution(s)
    mh_options: dict
        + 'type_prop': {'random walk', 'independent'}
          type of proposal: either Gaussian random walk, or independent Gaussian
        + 'adaptive': bool
          If True, the covariance matrix of the random walk proposal is
          set to a `rw_scale` times the weighted cov matrix of the particle
          sample (ignored if proposal is independent)
        + 'rw_scale': float (default=None)
          see above (ignored if proposal is independent)
        + 'indep_scale': float (default=1.1)
          for an independent proposal, the proposal distribution is
          Gaussian with mean set to the particle mean, cov set to
          `indep_scale` times particle covariance
        + 'nsteps': int (default: 0)
          number of steps; if 0, the number of steps is chosen adaptively
          as follows: we stop when the average distance between the
          starting points and the stopping points increase less than a
          certain fraction
        + 'delta_dist': float (default: 0.1)
          threshold for when nsteps = 0
    """

    def __init__(self, model, mh_options=None):
        self.model = model
        self.mh_options = {} if mh_options is None else mh_options

    @property
    def T(self):
        return self.model.T

    def default_moments(self, W, x):
        return rs.wmean_and_var_str_array(W, x.theta)

    def summary_format(self, smc):
        if smc.rs_flag:
            ars = np.array(smc.X.acc_rates[-1])
            to_add = ', Metropolis acc. rate (over %i steps): %.3f' % (
                ars.size, ars.mean())
        else:
            to_add = ''
        return 't=%i%s, ESS=%.2f' % (smc.t, to_add, smc.wgts.ESS)


class IBIS(FKSMCsampler):
    """FeynmanKac class for IBIS algorithm.

    see base class `FKSMCsampler` for parameters.
    """
    mutate_only_after_resampling = True  # override default value of FKclass

    def logG(self, t, xp, x):
        lpyt = self.model.logpyt(x.theta, t)
        x.lpost += lpyt
        return lpyt

    def compute_post(self, x, t):
        x.lpost = self.model.logpost(x.theta, t=t)

    def M0(self, N):
        x0 = MetroParticles(theta=self.model.prior.rvs(size=N))
        self.compute_post(x0, 0)
        return x0

    def M(self, t, Xp):
        # in IBIS, M_t leaves invariant p(theta|y_{0:t-1})
        comp_target = lambda x: self.compute_post(x, t-1)
        return Xp.Metropolis(comp_target, mh_options=self.mh_options)


class TemperingParticles(MetroParticles):
    shared = ['acc_rates', 'path_sampling']

    def __init__(self, theta=None, lprior=None, llik=None,
                 lpost=None, acc_rates=None, path_sampling=None):
        MetroParticles.__init__(self, theta=theta, lprior=lprior, llik=llik,
                                lpost=lpost, acc_rates=acc_rates)
        self.path_sampling = [0.] if path_sampling is None else path_sampling

class AdaptiveTemperingParticles(TemperingParticles):
    shared = ['acc_rates', 'exponents', 'path_sampling']

    def __init__(self, theta=None, lprior=None, llik=None,
                 lpost=None, acc_rates=None, exponents=None,
                 path_sampling=None):
        TemperingParticles.__init__(self, theta=theta, lprior=lprior, llik=llik,
                                    lpost=lpost, acc_rates=acc_rates,
                                    path_sampling=path_sampling)
        self.exponents = [0.] if exponents is None else exponents


class Tempering(FKSMCsampler):
    """FeynmanKac class for tempering SMC.

    Parameters
    ----------
    exponents: list-like
        Tempering exponents (must starts with 0., and ends with 1.)

    See base class for other parameters.
    """
    def __init__(self, model, mh_options=None, exponents=None):
        FKSMCsampler.__init__(self, model, mh_options=mh_options)
        self.exponents = exponents
        self.beta_diff = np.diff(exponents)
        self.q = None # set by subclass

    @property
    def T(self):
        return self.beta_diff.shape[0]

    def logG_tempering(self, x, new_beta, beta, new_q, q):
        dl = qdiff(x.llik, new_beta, beta, new_q, q)
        x.lpost += dl
        return dl

    def compute_post(self, x, epn, q=1):
        x.lprior = self.model.prior.logpdf(x.theta)
        x.llik = self.model.loglik(x.theta)
        x.lpost = qpath(x.lprior, x.lprior + x.llik, epn, self.q)

    def M0(self, N):
        x0 = TemperingParticles(theta=self.model.prior.rvs(size=N))
        self.compute_post(x0, 0.)
        return x0

    def M(self, t, Xp):
        epn = self.exponents[t]
        compute_target = lambda x: self.compute_post(x, epn, self.q)
        return Xp.Metropolis(compute_target, self.mh_options)


class AdaptiveTempering(Tempering):
    """Feynman-Kac class for adaptive tempering SMC.

    Parameters
    ----------
    ESSrmin: float
        Sequence of tempering dist's are chosen so that ESS ~ N * ESSrmin at
        each step

    See base class for other parameters.

    Note
    ----
    Since the successive temperatures are chosen so that the ESS
    drops to a certain value, it is highly recommended that you
    set option ESSrmin in SMC to 1., so that resampling is triggered
    at every iteration.
    """

    def __init__(self, model, args, mh_options=None, ESSrmin=0.5, delta=100):
        FKSMCsampler.__init__(self, model, mh_options=mh_options)
        self.ESSrmin = ESSrmin
        self.delta = delta
        self.q = delta_to_q(delta)
        self.deltahist = [delta]
        self.args = args
        self.fudge = 1.e-12

    def done(self, smc):
        if smc.X is None:
            return False  # We have not even started yet
        else:
            return smc.X.exponents[-1] >= 1.

    def logG(self, t, xp, x):
        beta = x.exponents[-1]
        ESSmin = self.ESSrmin * x.N
        new_q, old_q = self.q, self.q
        if ('q_init' in self.args.adaptive) and (t == 0):
            _, new_delta = learn_delta(x.llik, beta, ESSmin, self.fudge, restarts=self.args.q_init_restarts)
            print(f"new_delta: {new_delta}")
            self.delta = new_delta
            self.deltahist.append(new_delta)
            self.q = delta_to_q(new_delta)


        if 'fixed' in self.args.adaptive:
            new_beta = self.args.schedule[t]
        else:
            f = lambda e: rs.essl(qdiff(x.llik, e, beta)) - ESSmin
            # stopping critera
            if np.sign(f(beta)) == np.sign(f(1 - self.fudge)):
                # we're done (last iteration)
                new_beta = 1. # put 1. manually so that we can safely test == 1.
            else:
                new_beta = optimize.brentq(f, beta, 1 - self.fudge)  # secant search


        x.exponents.append(new_beta)
        return self.logG_tempering(x, new_beta, beta, new_q, old_q)


    def M0(self, N):
        x0 = AdaptiveTemperingParticles(theta=self.model.prior.rvs(size=N))
        self.compute_post(x0, 0., self.q)
        return x0

    def M(self, t, Xp):
        epn = Xp.exponents[-1]
        compute_target = lambda x: self.compute_post(x, epn, self.q)
        return Xp.Metropolis(compute_target, self.mh_options)

    def summary_format(self, smc):
        msg = FKSMCsampler.summary_format(self, smc)
        return msg + ', tempering exponent=%.3g' % smc.X.exponents[-1]


#####################################
# SMC^2

def rec_to_dict(arr):
    """ Turns record array *arr* into a dict """

    return dict(zip(arr.dtype.names, arr))


class ThetaWithPFsParticles(MetroParticles):
    """ class for a SMC^2 particle system """
    shared = ['acc_rates', 'just_moved', 'Nxs']

    def __init__(self, theta=None, lpost=None, acc_rates=None, pfs=None,
                 just_moved=False, Nxs=None):
        if pfs is None:
            pfs = FancyList([])
        if Nxs is None:
            Nxs = []
        MetroParticles.__init__(self, theta=theta, lpost=lpost, pfs=pfs,
                                acc_rates=acc_rates, just_moved=just_moved,
                                Nxs=Nxs)

    @property
    def Nx(self):  # for cases where Nx vary over time
        return self.pfs[0].N


class SMC2(FKSMCsampler):
    """ Feynman-Kac subclass for the SMC^2 algorithm.

    Parameters
    ----------
    ssm_cls: `StateSpaceModel` subclass
        the considered parametric state-space model
    prior: `StructDist` object
        the prior
    data: list-like
        the data
    smc_options: dict
        options to be passed to each SMC algorithm
    fk_cls: Feynman-Kac class (default: Bootstrap)
    mh_options: dict
        options for the Metropolis steps
    init_Nx: int
        initial value for N_x
    ar_to_increase_Nx: float
        Nx is increased (using an exchange step) each time
        the acceptance rate is above this value (if negative, Nx stays
        constant)
    """
    mutate_only_after_resampling = True  # override default value of FKclass

    def __init__(self, ssm_cls=None, prior=None, data=None, smc_options=None,
                 fk_cls=None, mh_options=None, init_Nx=100, ar_to_increase_Nx=-1.):
        FKSMCsampler.__init__(self, None, mh_options=mh_options)
        # switch off collection of basic summaries (takes too much memory)
        self.smc_options = {'summaries': False}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        self.fk_cls = Bootstrap if fk_cls is None else fk_cls
        if 'model' in self.smc_options or 'data' in self.smc_options:
            raise ValueError(
                'SMC2: options model and data are not allowed in smc_options')
        for k in ['ssm_cls', 'prior', 'data', 'init_Nx', 'ar_to_increase_Nx']:
            self.__dict__[k] = locals()[k]

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def logG(self, t, xp, x):
        # exchange step (should occur only immediately after a move step)
        we_increase_Nx = (
            x.just_moved and np.mean(x.acc_rates[-1]) < self.ar_to_increase_Nx)
        if we_increase_Nx:
            liw_Nx = self.exchange_step(x, t, 2 * x.Nx)
            x.just_moved = False
        # compute (estimate of) log p(y_t|\theta,y_{0:t-1})
        lpyt = np.empty(shape=x.N)
        for m, pf in enumerate(x.pfs):
            next(pf)
            lpyt[m] = pf.loglt
        x.lpost += lpyt
        x.Nxs.append(x.Nx)
        if we_increase_Nx:
            return lpyt + liw_Nx
        else:
            return lpyt

    def alg_instance(self, theta, N):
        return particles.SMC(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                            data=self.data),
                          N=N, **self.smc_options)

    def compute_post(self, x, t, Nx):
        x.pfs = FancyList([self.alg_instance(rec_to_dict(theta), Nx) for theta in
                           x.theta])
        x.lpost = self.prior.logpdf(x.theta)
        is_finite = np.isfinite(x.lpost)
        if t >= 0:
            for m, pf in enumerate(x.pfs):
                if is_finite[m]:
                    for _ in range(t + 1):
                        next(pf)
                    x.lpost[m] += pf.logLt

    def M0(self, N):
        x0 = ThetaWithPFsParticles(theta=self.prior.rvs(size=N))
        self.compute_post(x0, -1, self.init_Nx)
        return x0

    def M(self, t, xp):
        # Like in IBIS, M_t leaves invariant theta | y_{0:t-1}
        comp_target = lambda x: self.compute_post(x, t-1, xp.Nx)
        out = xp.Metropolis(comp_target, mh_options=self.mh_options)
        out.just_moved = True
        return out

    def exchange_step(self, x, t, new_Nx):
        old_lpost = x.lpost.copy()
        # exchange step occurs at beginning of step t, so y_t not processed yet
        self.compute_post(x, t - 1, new_Nx)
        return x.lpost - old_lpost

    def summary_format(self, smc):
        msg = FKSMCsampler.summary_format(self, smc)
        return msg + ', Nx=%i' % smc.X.Nx
