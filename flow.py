import torch
from torch import nn
from torch.distributions import Normal


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, bijections):
        super(MNFLinear, self).__init__()
        self.linear_f = nn.Linear(in_features, out_features)
        self.act_f = nn.GELU()
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
        activations = self.act_f(self.linear_f(input))
        latent_mean = torch.matmul(activations * aux, self.mean)
        latent_variance = torch.matmul(activations ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise


class MNFConv(nn.Module):
    pass
