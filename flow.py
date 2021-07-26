import torch
from torch import nn
from torch.distributions import Normal, Bernoulli


class RealNVP(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super(RealNVP, self).__init__()
        self.mask = None
        self.fc_linear = nn.Linear(in_features, hid_features)
        self.mean_linear = nn.Linear(hid_features, out_features)
        self.variance_linear = nn.Linear(hid_features, out_features)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def net(self, input, mask):
        activations = self.tanh(self.fc_linear(input * mask))
        mean = self.mean_linear(activations)
        variance = self.sigmoid(self.variance_linear(activations))
        return mean, variance

    def forward(self, input: torch.Tensor):
        mask = Bernoulli(.5).sample(input.shape)
        self.mask = mask
        mean, variance = self.net(input, mask)
        out = mask * input + (1 - mask) * (input * variance + (1 - variance) * mean)
        log_det_jacobian = torch.matmul((1 - mask), torch.log(variance))
        return out, log_det_jacobian

    def inverse(self, input: torch.Tensor):
        assert self.mask is not None
        assert input.shape == self.mask.shape
        mask = self.mask
        mean, variance = self.net(input, self.mask)
        out = input + (1 - mask) * (1 - variance) * mean
        out /= mask + (1 - mask) * variance
        return out


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, bijections):
        super(MNFLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_func = nn.GELU()
        self.mean = torch.zeros((out_features, out_features), requires_grad=True)
        self.variance = torch.tensor(torch.diag(out_features, out_features), requires_grad=True)
        self.bijections = bijections

    @property
    def base_dist(self):
        return Normal(0., 1.)

    @property
    def noise_dist(self):
        return Normal(0., 1.)

    def log_prob(self, input):
        log_prob = torch.zeros(input.shape[0])
        for func in self.bijections:
            input, log_det_jacobian = func(input)
            log_prob += log_det_jacobian
        return log_prob + self.base_dist.log_prob(input).sum(1)

    def sample_auxiliary(self):
        aux = self.base_dist.sample()
        for func in reversed(self.bijections):
            aux = func.inverse(aux)
        return aux

    def forward(self, input):
        aux = self.sample_auxiliary()
        activations = self.act_func(self.linear(input))
        latent_mean = torch.matmul(activations * aux, self.mean)
        latent_variance = torch.matmul(activations ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise


class MNFConv(nn.Module):
    pass
