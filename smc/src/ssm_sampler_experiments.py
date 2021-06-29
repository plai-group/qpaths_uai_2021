#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical experiment of Chapter 17 (SMC samplers).

Compare IBIS and SMC tempering for approximating:

* the normalising constant (marginal likelihood)
* the posterior expectation of the p coefficients

for a logistic regression model.

See below for how to select the data-set.
"""

from itertools import product
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy import random
import ml_helpers as mlh
import seaborn as sb
import pickle
from src import particles
from src.particles import datasets as dts
from src.particles import distributions as dists
from src.particles import resampling as rs
from src.particles import smc_samplers

from parallel import pmap

datasets = {'pima': dts.Pima, 'eeg': dts.Eeg, 'sonar': dts.Sonar}


def run(args):
    data = datasets[args.dataset_name]().data
    T, p = data.shape

    ESSrmin      = args.ESSrmin
    alg_type     = args.alg_type
    dataset_name = args.dataset_name
    Ms           = args.Ms
    nruns        = args.nruns
    typM         = args.typM
    N            = args.N

    # prior & model
    prior = dists.StructDist({'beta': dists.MvNormal(scale=5., cov=np.eye(p))})

    class LogisticRegression(smc_samplers.StaticModel):
        def logpyt(self, theta, t):
            # log-likelihood factor t, for given theta
            lin = np.matmul(theta['beta'], self.data[t, :])
            return - np.logaddexp(0., -lin)

        def loglik(self, theta, t=None):
            if t is None:
                # vectorized
                lin = self.data @ theta['beta'].T
                out = - np.logaddexp(0., -lin).sum(0)
                return out

            l = np.zeros(shape=theta.shape[0])
            for s in range(t + 1):
                l += self.logpyt(theta, s)

            return l


    def _inner(M, i, delta=100):
        data_copy = data.copy()
        # hack to fix weird seed bug which occurs while running in parallel
        seed  = int(M * i * delta * args.seed)
        mlh.seed_all(seed)
        if args.subsample != 0:
             data_copy = pd.DataFrame(data_copy).sample(int(T * args.subsample)).values

        # need to shuffle the data for IBIS
        random.shuffle(data_copy)
        model = LogisticRegression(data=data_copy, prior=prior)

        fk = smc_samplers.AdaptiveTempering(
            model,
            args,
            ESSrmin=ESSrmin,
            mh_options={'nsteps': M},
            delta=delta)
        pf = particles.SMC(N=N, fk=fk, ESSrmin=1., moments=True, verbose=True)

        print('%s, M=%i, run %i' % (alg_type, M, i))
        pf.run()
        print('CPU time (min): %.2f' % (pf.cpu_time / 60))
        print('loglik: %f' % pf.logLt)
        res = {'M': M, 'type': alg_type,
               'out': pf.summaries, 'cpu': pf.cpu_time}

        n_eval = N * T * (1. + M * (len(pf.summaries.ESSs) - 1))

        res['path_sampling'] = pf.X.path_sampling[-1]
        res['exponents'] = pf.X.exponents
        res['q'] = pf.fk.q
        res['delta'] = pf.fk.delta
        res['n_eval'] = n_eval
        res['sample_logliks'] = model.loglik(prior.rvs(10))
        res['seed'] = seed
        return res

    print('Dataset: %s' % dataset_name)
    results = pmap(
        lambda M_i_alg: _inner(*M_i_alg), product(Ms, range(nruns), args.deltas),
        n_jobs=args.n_jobs
        )

    return {'results': results,
            'dataset_name': dataset_name,
            'Ms': Ms,
            'typM': typM,
            'N': N,
            'nruns': nruns,
            'T': T,
            'p': p,
            'ESSrmin': ESSrmin,
            'subsample': args.subsample,
            'data': data}
