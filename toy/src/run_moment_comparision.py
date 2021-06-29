from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.ais import Student1D, run_ais, Gaussian1D, moment_average, geometric_average
from src.paths import  q_moment, qpath
from src.utils import pmap


def compute_bounds_ais( proposal, target, no_betas, args, path="qpath", q=1.0):
    beta_vec = np.linspace(0, 1.0, no_betas)

    if path == "moment":
        beta_dists = [moment_average(proposal, target, beta) for beta in beta_vec]
    elif path == "qmoment":
        beta_dists = [q_moment(proposal, target, beta) for beta in beta_vec]
    elif (path == "geometric") or ((q == 1.0) and (path == 'qpath')):
        beta_dists = [geometric_average(proposal, target, beta) for beta in beta_vec]
    elif path == "qpath":
        beta_dists = [qpath(proposal, target, beta, q) for beta in beta_vec]
    else:
        raise ValueError

    def _run(seed):
        np.random.seed(seed)
        logZ, logZ_lower, logZ_upper = run_ais(beta_dists, args.no_samples)
        res = {
            "no_betas":no_betas,
            "path":path,
            "q":q,
            "logZ":logZ,
            "logZ_lower":logZ_lower,
            "logZ_upper":logZ_upper
        }

        return res

    return pd.DataFrame([_run(args.seed)])


def compare_bounds(args):
    q_vec = args.q_vec
    no_betas = [args.no_betas]

    if args.mode == 'student':
        proposal = Student1D(q=2, mean=-4.0, variance=3)
        target   = Student1D(q=2, mean=4.0 , variance=1)
        moment_alg = ['qmoment']
    elif args.mode == 'gaus':
        proposal = Gaussian1D(mean=-4.0, variance=3)
        target   = Gaussian1D(mean=4.0 , variance=1)
        moment_alg = ['moment']
    else:
        raise ValueError

    def _run_qpath(no_betas_q):
        no_betas, q = no_betas_q
        return compute_bounds_ais(proposal, target, no_betas, args, q=q, path="qpath")

    def _run_moments(no_betas_path):
        no_betas, path = no_betas_path
        return compute_bounds_ais(proposal, target, no_betas, args, path=path)

    qpath_res = pmap(_run_qpath, product(no_betas, q_vec), n_jobs=args.n_jobs)
    qmoments = pmap(_run_moments, product(no_betas, moment_alg), n_jobs=args.n_jobs)
    df = pd.concat((qpath_res + qmoments))
    df['seed'] = args.seed
    df['no_samples'] = args.no_samples
    df['gap'] = df['logZ_upper'] - df['logZ_lower']

    return df
