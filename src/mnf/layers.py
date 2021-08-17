from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class MNFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, forward_bijections, backward_bijections, log_variance, mean):
        super(MNFLayer, self).__init__()
        self.tanh = nn.Tanh()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._aux0 = None
        self.c = nn.Parameter(torch.ones(in_channels))
        self.b1 = nn.Parameter(torch.ones(in_channels))
        self.b2 = nn.Parameter(torch.ones(in_channels))

        self.forward_bijections = forward_bijections
        self.backward_bijections = backward_bijections
        self.sigmoid = nn.Sigmoid()
        self._log_variance = nn.Parameter(log_variance)
        self.mean = nn.Parameter(mean)

    @property
    def variance(self):
        if not hasattr(self, '_log_variance'):
            raise NotImplementedError('MNFLayer must have _log_variance as an attribute')
        return self._log_variance.exp()

    @property
    def base_dist(self):
        return Normal(1., 1.)

    @property
    def noise_dist(self):
        return Normal(0., 1.)

    def push_forward(self, aux0, sample_mask=True):
        self._aux0 = aux0
        aux = aux0
        log_det_tot = 0.
        for func in self.forward_bijections:
            aux, log_det = func(aux, sample_mask)
            log_det_tot += log_det
        return aux, log_det

    def push_back(self, aux, sample_mask=True):
        aux_back = aux
        log_det_tot = 0.
        for func in reversed(self.backward_bijections):
            aux_back, log_det = func(aux_back, sample_mask)
            log_det_tot += log_det
        return aux_back, log_det_tot

    def log_prob_aux_back(self, aux_back):
        mean = torch.permute(self.mean, (-1, -2)).reshape((-1, self.in_channels))
        variance = torch.permute(self.variance, (-1, -2)).reshape((-1, self.in_channels))
        out_channels = torch.numel(mean[..., 0])

        noise = self.base_dist.sample((out_channels,))
        aux, _ = self.push_forward(self._aux0, sample_mask=False)

        cw = self.tanh(
            torch.matmul(mean, self.c * aux) + torch.sqrt(torch.matmul(variance, self.c ** 2))) * noise

        loc = torch.matmul(1 / out_channels * torch.ones(out_channels), torch.outer(cw, self.b1))
        std = torch.sigmoid(torch.matmul(1 / out_channels * torch.ones(out_channels), torch.outer(cw, self.b2)))

        return torch.distributions.Normal(loc, std).log_prob(aux_back)

    def kl_divergence(self):
        aux, log_det_aux = self.push_forward(self._aux0, sample_mask=False)
        log_prob_aux = self.base_dist.log_prob(self._aux0).sum() - log_det_aux

        mean = self.mean.reshape((-1, self.in_channels, self.out_channels))
        variance = self.variance.reshape((-1, self.in_channels, self.out_channels))

        kl_qw_pw = variance + (aux.reshape((1, -1, 1)) * mean) ** 2 - torch.log(variance) - 1

        aux_back, log_det_aux_back = self.push_back(aux)
        log_prob_aux_back = self.log_prob_aux_back(aux_back) + log_det_aux_back

        return -.5 * kl_qw_pw.sum() + log_prob_aux_back.sum() - log_prob_aux.sum()


class MSFLayer(MNFLayer):
    def __init__(self, in_channels, out_channels, bijections, log_variance, mean):
        super(MSFLayer, self).__init__(in_channels,
                                       out_channels,
                                       bijections,
                                       bijections,
                                       log_variance,
                                       mean)

    def kl_divergence(self):
        aux, log_det_aux = self.push_forward(self._aux0, sample_mask=False)
        log_prob_aux = self.base_dist.log_prob(self._aux0).sum() + log_det_aux

        mean = self.mean.reshape((-1, self.in_channels, self.out_channels))
        variance = self.variance.reshape((-1, self.in_channels, self.out_channels))

        kl_qw_pw = variance + (aux.reshape((1, -1, 1)) * mean) ** 2 - torch.log(variance) - 1

        aux_back, log_det_aux_back = self.push_back(aux)
        log_prob_aux_back = self.base_dist.log_prob(aux_back).sum() + log_det_aux_back

        return -.5 * kl_qw_pw.sum() + log_prob_aux_back.sum() - log_prob_aux.sum()

    def push_back(self, aux, sample_mask=False):
        aux_back = aux
        log_det_tot = 0.
        for func in reversed(self.backward_bijections):
            aux_back, log_det = func.pullback(aux_back, sample_mask)
            log_det_tot += log_det
        return aux_back, log_det_tot


class MSFLinear(MSFLayer):
    def __init__(self, in_channels, out_channels, bijections):
        mean = nn.Parameter(torch.empty((in_channels, out_channels), requires_grad=True))
        nn.init.xavier_normal_(mean)
        log_variance = torch.distributions.Normal(torch.exp(torch.tensor(-9)), 1e-3).sample((in_channels, out_channels))
        super(MSFLinear, self).__init__(in_channels, out_channels,
                                        bijections,
                                        log_variance, mean)

    def forward(self, features):
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.in_channels,)))
        latent_mean = torch.matmul(aux * features, self.mean)
        latent_variance = torch.matmul(features ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise


class MNFConv2D(MSFLayer):
    def __init__(self, in_channels, out_channels, kernel_size: Tuple, bijections, stride=1):
        mean = torch.empty((*kernel_size, in_channels, out_channels), requires_grad=True)
        nn.init.xavier_normal_(mean)
        log_variance = nn.Parameter(
            torch.distributions.Normal(torch.exp(torch.tensor(-9)), 1e-3).sample(
                (*kernel_size, in_channels, out_channels)))

        super(MNFConv2D, self).__init__(in_channels, out_channels,
                                        bijections, log_variance, mean)
        self.stride = stride

    def forward(self, features):
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.in_features,)))
        latent_mean = torch.conv2d(features, self.mean * aux.reshape((1, 1, -1, 1)), stride=self.stride)
        latent_variance = torch.conv2d(features ** 2, self.variance, stride=self.stride)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise


class MNFLinear(MNFLayer):
    def __init__(self, in_channels, out_channels, forward_bijections, backward_bijections):
        mean = nn.Parameter(torch.empty((in_channels, out_channels), requires_grad=True))
        nn.init.xavier_normal_(mean)
        log_variance = torch.distributions.Normal(torch.exp(torch.tensor(-9)), 1e-3).sample((in_channels, out_channels))
        super(MNFLinear, self).__init__(in_channels, out_channels,
                                        forward_bijections, backward_bijections,
                                        log_variance, mean)

    def forward(self, features):
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.in_channels,)))
        latent_mean = torch.matmul(aux * features, self.mean)
        latent_variance = torch.matmul(features ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise


class MNFConv2D(MNFLayer):
    def __init__(self, in_channels, out_channels, kernel_size: Tuple, forward_bijections, backward_bijections,
                 stride=1):
        mean = torch.empty((*kernel_size, in_channels, out_channels), requires_grad=True)
        nn.init.xavier_normal_(mean)
        log_variance = nn.Parameter(
            torch.distributions.Normal(torch.exp(torch.tensor(-9)), 1e-3).sample(
                (*kernel_size, in_channels, out_channels)))

        super(MNFConv2D, self).__init__(in_channels, out_channels,
                                        forward_bijections, backward_bijections,
                                        log_variance, mean)
        self.stride = stride

    def forward(self, features):
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.in_features,)))
        latent_mean = torch.conv2d(features, self.mean * aux.reshape((1, 1, -1, 1)), stride=self.stride)
        latent_variance = torch.conv2d(features ** 2, self.variance, stride=self.stride)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise
