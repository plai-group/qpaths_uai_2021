import argparse
import numpy as np
import itertools
import torch
import ais
import dgm


parser = argparse.ArgumentParser(description="AIS with a trained VAE model")
parser.add_argument(
    "--latent-dim",
    type=int,
    default=50,
    metavar="D",
    help="number of latent variables (default: 50)",
)
parser.add_argument(
    "--hidden-dim",
    type=int,
    default=200,
    metavar="H",
    help="number of hidden dimensions (default: 200)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=10,
    metavar="N",
    help="number of examples to eval at once (default: 10)",
)
parser.add_argument(
    "--no-batches",
    type=int,
    default=10,
    metavar="B",
    help="number of batches for simulation/evaluation (default: 10)",
)
parser.add_argument(
    "--chain-length",
    type=int,
    default=500,
    metavar="L",
    help="length of ais chain (default: 500)",
)
parser.add_argument(
    "--no-samples",
    type=int,
    default=100,
    metavar="I",
    help="number of ais samples (default: 100)",
)
parser.add_argument(
    "--model-path",
    type=str,
    default="model_files/model.pt",
    metavar="C",
    help="path to checkpoint",
)
parser.add_argument(
    "--q",
    type=float,
    default=0.8,
    metavar="Q",
    help="q value for q-path",
)
parser.add_argument(
    "--qpath", dest="use_qpath", action="store_true", help="use q-path"
)
parser.add_argument(
    "--geometric",
    dest="use_qpath",
    action="store_false",
    help="use geometric path",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    help="a quick run to test the set-up",
)

parser.set_defaults(use_qpath=False)

args = parser.parse_args()


def run_bdmc(model, loader, beta_schedule, n_sample, use_qpath, q, use_cuda):
    """Bidirectional Monte Carlo with Annealed Importance Sampling

    Args:
      model (vae.VAE): VAE model
      loader (iterator): iterator to loop over pairs of Variables; the first
        entry being `x`, the second is not used
      beta_schedule (list or numpy.ndarray): forward temperature schedule
      use_cuda (bool): whether to use a GPU
    Returns:
        Lists for forward bound on batchs of data
    """

    # iterator is exhaustable in py3, so need duplicate
    loader_1, loader_2 = itertools.tee(loader, 2)

    # forward chain
    print("running forward ais...")
    forward_schedule = beta_schedule
    forward_logws = ais.ais_trajectory(
        model,
        loader_1,
        schedule=forward_schedule,
        n_sample=n_sample,
        use_qpath=use_qpath,
        q=q,
        use_cuda=use_cuda,
    )

    # backward chain
    print("running reverse ais...")
    backward_schedule = np.flip(forward_schedule, axis=0)
    backward_logws = ais.ais_trajectory(
        model,
        loader_2,
        schedule=backward_schedule,
        n_sample=n_sample,
        use_qpath=use_qpath,
        q=q,
        use_cuda=use_cuda,
        forward=False,
    )

    lower_bounds = []
    upper_bounds = []

    for i, (forward, backward) in enumerate(zip(forward_logws, backward_logws)):
        lower_bounds.append(forward.mean().detach().item())
        upper_bounds.append(backward.mean().detach().item())

    lower_bound_mean = np.mean(lower_bounds)
    upper_bound_mean = np.mean(upper_bounds)

    print(
        "Average bound on simulated data: lower %.4f, upper %.4f"
        % (lower_bound_mean, upper_bound_mean)
    )

    return (
        forward_logws,
        backward_logws,
        lower_bound_mean,
        upper_bound_mean,
    )


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load trained model from a checkpoint
    use_cuda = torch.cuda.is_available()
    model = dgm.VAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    model_path = args.model_path
    if use_cuda:
        model.cuda()
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    if args.dry_run:
        no_samples = 10
        chain_length = 50
        no_batches = 2
        batch_size = 50
    else:
        no_samples = args.no_samples
        chain_length = args.chain_length
        no_batches = args.no_batches
        batch_size = args.batch_size

    # simulate data
    loader = model.sample_data(batch_size, no_batches, use_cuda)

    # run ais with a linear schedule
    beta_schedule = np.linspace(0.0, 1.0, chain_length)
    res_lb, res_ub, average_lb, average_ub = run_bdmc(
        model,
        loader,
        beta_schedule=beta_schedule,
        n_sample=no_samples,
        use_qpath=args.use_qpath,
        q=args.q,
        use_cuda=use_cuda,
    )
    print(
        "average lower bound %.3f, average upper bound %.3f, average bdmc gap: %.3f"
        % (average_lb, average_ub, average_ub - average_lb)
    )


if __name__ == "__main__":
    main()


# python run_bdmc_dgm.py --batch-size=$batchsize --no-batches=$nobatches --chain-length=${chainlength[$index]} --no-samples=${nosamples[$index]} --geometric --seed $seed --model-path model_files/tvo_model_h_200_z_50_T_100_scdhl_log_lbmin_-1.5_bs_100.pt
