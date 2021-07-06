import os

import sys
import pandas as pd
import numpy as np
import flavor
from ml_helpers import default_init, join_path
from src import assertions, ssm_sampler_experiments
import ml_helpers as mlh
from sacred import Experiment
import wandb

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking

ex = Experiment()
WANDB_PROJECT_NAME = 'my_project_name'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'

GT = {
    'pima': -391.4943977001596,
    'sonar': -123.66725198734291,
}

@ex.config
def my_config():
    # paths
    home_dir = '.' # required by job submitter, don't modify
    artifact_dir = './artifacts/' # required by job submitter, don't modify
    dataset_name = 'pima' # 'eeg', 'sonar'

    adaptive = 'beta' # ['fixed', 'beta', 'q_init', 'q_init_fixed_beta']
    alg_type = 'qpath' # 'ground_truth'

    n_jobs = -1

    ESSrmin = 0.5
    nruns = 10
    seed = 1

    K = 10

    q_init_samples  = 10000
    q_init_restarts = 500

    N = 10 ** 4
    Ms = [1, 3, 5]
    typM = 5

    # q = 1 - 10**(-delta)
    deltas = [20]

    subsample = 0

def convert_to_df(dp, args):
    results_df = (pd
     .DataFrame(dp['results'])
     .assign(
         last_loglik    = lambda x: x.out.apply(lambda y: y.logLts[-1]),
         first_post_exp = lambda x: x.out.apply(lambda y: y.moments[-1]['mean']['beta'][1]),
         moments        = lambda x: x.out.apply(lambda y: y.moments[-1]),
         beta_mean      = lambda x: x.moments.apply(lambda y:y['mean']['beta']),
         beta_var       = lambda x: x.moments.apply(lambda y:y['var']['beta']),
     )
     .add_column('ESSrmin', dp['ESSrmin'])
     .add_column('T', dp['T'])
     .add_column('p', dp['p'])
     .add_column('N', dp['N'])
     .add_column('dataset_name', dp['dataset_name'])
     .add_column('nruns', dp['nruns'])
     .expand_list_column('beta_mean', [f'mean_{i}' for i in range(dp['p'])])
     .expand_list_column('beta_var', [f'var_{i}' for i in range(dp['p'])]))


    results_df['adaptive']        = args.adaptive
    results_df['alg_type']        = args.alg_type
    results_df['q_init_restarts'] = args.q_init_restarts

    return results_df



def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = default_init(config)
    args.artifact_dir = join_path(args.home_dir, args.artifact_dir)
    assertions.validate_hypers(args)
    mlh.seed_all(args.seed)

    if args.alg_type == 'ground_truth':
        args.N = 50000
        args.adaptive = 'beta'
        args.alg_type = 'qpath'
        args.nruns = 10
        args.Ms = [20]

    # fixed for now
    args.schedule = np.linspace(0, 1, args.K)[1:]

    return args


# Main training loop
def train(args):
    res = ssm_sampler_experiments.run(args)
    df = convert_to_df(res, args)
    df.to_pickle(f"{wandb.run.dir}/df.pkl")
    return df

@ex.automain
def command_line_entry(_run,_config):
    wandb_run = wandb.init(project = WANDB_PROJECT_NAME,
                            config = _config,
                              tags = [_run.experiment_info['name']])
    args = init(_config)
    train(args)
