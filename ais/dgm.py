import torch
import torch.nn as nn

from third.BDMC import utils
from torch.distributions import Bernoulli


class VAE(nn.Module):
    def __init__(self, latent_dim=50, hidden_dim=200):
        nn.Module.__init__(self)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        x_dim = 784

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_sig = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, net):
        h = self.encoder(net)
        mu, _std = self.enc_mu(h), self.enc_sig(h)
        std = nn.functional.softplus(_std)
        logvar = 2 * torch.log(std)
        return mu, logvar

    def decode(self, net):
        return self.decoder(net)

    def log_prior(self, z, use_cuda):
        zeros = torch.zeros(z.shape[0], self.latent_dim)
        if use_cuda:
            zeros = zeros.cuda()
        res = utils.log_normal(z, zeros, zeros)
        return res

    def log_likelihood(self, z, data):
        res = utils.log_bernoulli(self.decode(z), data)
        return res

    def log_joint(self, z, data, use_cuda):
        return self.log_prior(z, use_cuda) + self.log_likelihood(z, data)

    def sample_data(self, batch_size=10, n_batch=1, use_cuda=False):
        """Simulate data from the VAE model. Sample from the
        joint distribution p(z)p(x|z). This is equivalent to
        sampling from p(x)p(z|x), i.e. z is from the posterior.

        Args:
            model: VAE model for simulation
            batch_size: batch size for simulated data
            n_batch: number of batches

        Returns:
            iterator that loops over batches of torch Tensor pair x, z
        """

        batches = []
        for i in range(n_batch):
            # assume prior for VAE is unit Gaussian
            z = torch.randn(batch_size, self.latent_dim)
            if use_cuda:
                z = z.cuda()
            x_logits = self.decode(z)
            if isinstance(x_logits, tuple):
                x_logits = x_logits[0]
            x_bernoulli_dist = Bernoulli(probs=x_logits.sigmoid())
            x = x_bernoulli_dist.sample().data

            paired_batch = (x, z)
            batches.append(paired_batch)

        return iter(batches)
