import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import grad as torchgrad
from third.BDMC import hmc
from third.BDMC import utils


def log_sum_weighted_exp(val1, val2, weight1, weight2):
    """This is like log sum exp but with two weighted arguments

    Args:
        val1: first value inside log sum exp
        val2: second value inside log sum exp
        weight1: weight of first value
        weight2: weight of second value

    Returns:
        float: log of (weight1 * exp(val1) + weight2 * exp(val2))
    """
    val_max = torch.where(val1 > val2, val1, val2)
    val1_exp = weight1 * torch.exp(val1 - val_max)
    val2_exp = weight2 * torch.exp(val2 - val_max)
    lse = val_max + torch.log(val1_exp + val2_exp)
    return lse


def log_f_i_geometric(model, z, data, beta, use_cuda):
    """Compute log unnormalised density for intermediate distribution `f_i`
        using a geometric mixture:
        f_i = p(z)^(1-beta) p(x,z)^(beta) = p(z) p(x|z)^beta
        => log f_i = log p(z) + beta * log p(x|z)

    Args:
        model: model that can invoke log prior and log likelihood
        z: latent value at which to evaluate log density
        data: data to compute the log likelihood
        beta: mixture weight
        use_cuda: whether using cuda or not

    Returns:
        log unnormalised density
    """

    log_prior = model.log_prior(z, use_cuda)
    log_likelihood = model.log_likelihood(z, data)
    log_joint = log_prior + log_likelihood.mul_(beta)
    return log_joint


def log_f_i_qpath(model, z, data, beta, q, use_cuda):
    """Compute log unnormalised density for intermediate distribution `f_i`
        using a q-mixture:
        f_i = ( (1-beta) p(z)^(1-q) + beta p(x,z)^(1-q) )^(1/(1-q))

    Args:
        model: model that can invoke log prior and log likelihood
        z: latent value at which to evaluate log density
        data: data to compute the log likelihood
        beta: mixture weight
        q: q value for the mixture
        use_cuda: whether using cuda or not

    Returns:
        log unnormalised density
    """
    log_prior = model.log_prior(z, use_cuda)
    log_likelihood = model.log_likelihood(z, data)
    log_a = log_prior
    log_b = log_prior + log_likelihood
    if beta == 0.0:
        log_joint = log_a
    elif beta == 1.0:
        log_joint = log_b
    else:
        log_a = (1 - q) * log_a
        log_b = (1 - q) * log_b
        lse = log_sum_weighted_exp(log_a, log_b, 1 - beta, beta)
        log_joint = lse / (1 - q)
    return log_joint


def log_f_i(model, z, data, beta, use_qpath=False, q=0.8, use_cuda=False):
    """Wrapper function, return log_f_i_geometric or log_f_i_qpath"""
    if use_qpath:
        return log_f_i_qpath(model, z, data, beta, q, use_cuda)
    else:
        return log_f_i_geometric(model, z, data, beta, use_cuda)


def ais_trajectory(
    model,
    loader,
    schedule=np.linspace(0.0, 1.0, 500),
    n_sample=100,
    use_qpath=False,
    q=0.8,
    use_cuda=False,
    forward=True,
):
    """Compute annealed importance sampling trajectories for a batch of data.
    Could be used for *both* forward and reverse chain in BDMC.

    Args:
      model (vae.VAE): VAE model
      loader (iterator): iterator that returns pairs, with first component
        being `x`, second would be `z` or label (z will be used for backward ais)
      schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
      n_sample (int): number of importance samples
      use_qpath (bool): whether a q-path is used. If false, use geometric.
      q (float): q value for q-path
      use_cuda (bool): whether or not to use a GPU
      forward (bool): whether this is a forward or reverse AIS chain

    Returns:
        A list where each element is a torch.autograd.Variable that contains the
        log importance weights for a single batch of data
    """

    logws = []
    for i, (batch, post_z) in enumerate(loader):
        B = batch.size(0) * n_sample
        if use_cuda:
            batch = batch.cuda()
        batch = utils.safe_repeat(batch, n_sample)

        with torch.no_grad():
            if use_cuda:
                epsilon = torch.ones(B).cuda().mul_(0.01)
                accept_hist = torch.zeros(B).cuda()
                logw = torch.zeros(B).cuda()
            else:
                epsilon = torch.ones(B).mul_(0.01)
                accept_hist = torch.zeros(B)
                logw = torch.zeros(B)

        # initial sample of z
        if forward:
            current_z = torch.randn(B, model.latent_dim)
        else:
            current_z = utils.safe_repeat(post_z, n_sample)
        if use_cuda:
            current_z = current_z.cuda()
        current_z = current_z.requires_grad_()

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            # update log importance weight
            log_int_1 = log_f_i(
                model, current_z, batch, t0, use_qpath, q, use_cuda
            )
            log_int_2 = log_f_i(
                model, current_z, batch, t1, use_qpath, q, use_cuda
            )
            log_diff = log_int_2 - log_int_1
            logw += log_diff
            # resample velocity
            current_v = torch.randn(current_z.size())
            if use_cuda:
                current_v = current_v.cuda()

            def U(z):
                return -log_f_i(model, z, batch, t1, use_qpath, q, use_cuda)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B)
                if use_cuda:
                    grad_outputs = grad_outputs.cuda()
                # torch.autograd.grad default returns volatile
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                # clip by norm
                max_ = B * model.latent_dim * 100.0
                grad = torch.clamp(grad, -max_, max_)
                grad.requires_grad_()
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(B, model.latent_dim)
                if use_cuda:
                    zeros = zeros.cuda()
                return -utils.log_normal(v, zeros, zeros)

            z, v = hmc.hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
            current_z, epsilon, accept_hist = hmc.accept_reject(
                current_z,
                current_v,
                z,
                v,
                epsilon,
                accept_hist,
                j,
                U,
                K=normalized_kinetic,
                use_cuda=use_cuda,
            )
        logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        if not forward:
            logw = -logw
        logws.append(logw.detach().data)
        print("Last batch stats %.4f" % (logw.mean().cpu().data.numpy()))
    return logws
