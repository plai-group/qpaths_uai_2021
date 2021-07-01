This folder contains code for the SMC experiments shown in Figure 4 and Table 1.

### Acknowledgements

We use the [particles](https://github.com/nchopin/particles) package developed to accompany the textbook [An introduction to Sequential Monte Carlo](https://www.springer.com/gp/book/9783030478445). Additionally we use the machine learning project template available [here](https://github.com/vmasrani/ml_project_skeleton) and ML helper functions available [here](https://github.com/vmasrani/machine_learning_helpers).

### Set up

We use the following python packages:

`pip install pandas joblib pyjanitor tqdm torch matplotlib seaborn numba sacred wandb`

As well as the machine learning helper functions available at `https://github.com/vmasrani/machine_learning_helpers`.

```bash
python -m venv env
source env/bin/activate
git clone git@github.com:vmasrani/machine_learning_helpers.git
cd machine_learning_helpers/
export PYTHONPATH="$(pwd):${PYTHONPATH}"
cd ..
pip install pandas joblib pyjanitor tqdm torch matplotlib seaborn numba sacred wandb
python main.py with n_jobs=-1 adaptive=fixed_beta dataset_name=sonar K=20 --name 'q_heuristic' --unobserved
```

The structure of `main.py` is described in the comments [here](https://github.com/vmasrani/ml_project_skeleton/blob/master/main.py) and the command line interface is described [here](https://github.com/vmasrani/ml_project_skeleton/blob/master/README.md).

### Example

Jobs can be run from the command line via:

```bash
python main.py with n_jobs=-1 adaptive=fixed_beta dataset_name=sonar K=20 --name 'q_heuristic' --unobserved
```

To save results to wandb, drop the `--unobserved`.

