import sys
import os
from pathlib import Path
import numpy as np

import pandas as pd
from src.run_moment_comparision import compare_bounds

from ml_helpers import default_init, join_path
import ml_helpers as mlh
from sacred import Experiment
import wandb

# Use sacred for command line interface + hyperparams
# Use wandb for experiment tracking

ex = Experiment()
WANDB_PROJECT_NAME = 'qpath_toy'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'


@ex.config
def my_config():
    # paths
    home_dir = '.' # required by job submitter, don't modify
    artifact_dir = './artifacts/' # required by job submitter, don't modify

    seed = 1
    mode = 'student'
    no_samples = 2500
    no_betas   = 100
    n_jobs = -1

def init(config):
    # This gives dot access to all paths, hyperparameters, etc
    args = default_init(config)
    args.artifact_dir = join_path(args.home_dir, args.artifact_dir)
    mlh.seed_all(args.seed)

    args.q_vec = [0.9, 1, 2.0]

    return args

# Main training loop
def train(args):
    path = f"./res.pickle"
    df = compare_bounds(args)
    df.to_pickle(path)
    wandb.save(path)
    return df


# Don't touch this function!
@ex.automain
def command_line_entry(_run,_config):
    wandb_run = wandb.init(project = WANDB_PROJECT_NAME,
                            config = _config,
                              tags = [_run.experiment_info['name']])
    args = init(_config)
    train(args)
