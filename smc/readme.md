This folder contains code for the SMC experiments shown in Figure 4 and Table 1.

### Acknowledgements

We use the [particles](https://github.com/nchopin/particles) package developed to accompany the textbook [An introduction to Sequential Monte Carlo](https://www.springer.com/gp/book/9783030478445). Additionally we use the machine learning project template available [here](https://github.com/vmasrani/ml_project_skeleton) and ML helper functions available [here](https://github.com/vmasrani/machine_learning_helpers).

### Set up

Install the requirements using `pip install -r requirements.txt` and follow [the instructions](https://github.com/vmasrani/machine_learning_helpers) to enable `import ml_helpers as mlh` , `import flavor`, and `import parallel`. The structure of `main.py` is described in the comments [here](https://github.com/vmasrani/ml_project_skeleton/blob/master/main.py) and the command line interface is described [here](https://github.com/vmasrani/ml_project_skeleton/blob/master/README.md).

### Example

Jobs can be run from the command line via:

```bash
python main.py with n_jobs=-1 adaptive=fixed_beta dataset_name=sonar K=20 --name 'q_heuristic' --unobserved
```

To save results to wandb, drop the `--unobserved`.

