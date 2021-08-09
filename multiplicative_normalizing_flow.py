import math

import torch
from torch import nn
from torch.distributions import Normal

from utils import build_flows


class MNFFeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, hid_forward_flow, hid_inv_flow, num_flows):
        super().__init__()

        self.l1 = MNFLinear(in_channels, 128,
                            *build_flows(in_channels, hid_forward_flow, hid_inv_flow, num_flows))
        self.l2 = MNFLinear(128, 32, *build_flows(128, hid_forward_flow, hid_inv_flow, num_flows))
        self.l3 = MNFLinear(32, 8, *build_flows(32, hid_forward_flow, hid_inv_flow, num_flows))
        self.l4 = MNFLinear(8, 1, *build_flows(8, hid_forward_flow, hid_inv_flow, num_flows))

        self.layers = [self.l1, self.l2, self.l3, self.l4]
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.l4(x)
        return x

    def log_likelihood(self, x, y):
        return -.5 * torch.log(torch.tensor(2 * math.pi)) - .5 * (y - self(x)) ** 2

    def loss(self, x, y):
        return -(self.log_likelihood(x, y).sum() + sum(layer.kl_divergence() for layer in self.layers)).sum()


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, forward_bijections, backward_bijections):
        super(MNFLinear, self).__init__()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.mean = nn.Parameter(torch.empty((in_features, out_features), requires_grad=True))
        nn.init.xavier_normal_(self.mean)
        self.log_variance = nn.Parameter(
            torch.distributions.Normal(torch.exp(torch.tensor(-9)), 1e-3).sample((in_features, out_features)))

        self.forward_bijections = forward_bijections
        self.backward_bijections = backward_bijections

        self.inv_dout = 1 / out_features * torch.ones(out_features)
        self.c = nn.Parameter(torch.ones(in_features))
        self.b1 = nn.Parameter(torch.ones(in_features))
        self.b2 = nn.Parameter(torch.ones(in_features))

        self.in_features = in_features
        self.out_features = out_features

        self._aux0 = None

    @property
    def variance(self):
        return self.log_variance.exp()

    @property
    def base_dist(self):
        return Normal(0., 1.)

    @property
    def noise_dist(self):
        return Normal(0., 1.)

    def push_forward(self, aux0):
        self._aux0 = aux0
        aux = aux0
        log_det_tot = 0.
        for func in self.forward_bijections:
            aux, log_det = func(aux)
            log_det_tot += log_det
        return aux, log_det

    def push_back(self, aux):
        aux_back = aux
        log_det_tot = 0.
        for func in reversed(self.backward_bijections):
            aux_back, log_det = func(aux_back)
            log_det_tot += log_det
        return aux_back, log_det_tot

    def forward(self, features):
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.in_features,)))
        latent_mean = torch.matmul(aux * features, self.mean)
        latent_variance = torch.matmul(features ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise

    def log_prob_aux_back(self, aux_back):
        noise = self.base_dist.sample((self.out_features,))
        aux, _ = self.push_forward(self._aux0)
        cw = self.tanh(
            torch.matmul(self.c * aux, self.mean) + torch.sqrt(torch.matmul(self.c ** 2, self.variance)) * noise)

        loc = torch.matmul(torch.outer(self.b1, cw), self.inv_dout)
        std = self.sigmoid(torch.matmul(torch.outer(self.b2, cw), self.inv_dout))

        return torch.distributions.Normal(loc, std).log_prob(aux_back)

    def kl_divergence(self):
        aux, log_det_aux = self.push_forward(self._aux0)
        log_prob_aux = self.base_dist.log_prob(self._aux0) - log_det_aux

        kl_qw_pw = .5 * (self.variance + aux.unsqueeze(1) * self.mean + torch.log(self.variance))

        aux_back, log_det_aux_back = self.push_back(aux)
        log_prob_aux_back = self.log_prob_aux_back(aux_back) + log_det_aux_back

        return kl_qw_pw.sum() + log_prob_aux_back.sum() - log_prob_aux.sum()


class MNFConv(nn.Module):
    pass
