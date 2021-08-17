import torch
from torch import nn
from torch.distributions import Bernoulli


class RealNVP(nn.Module):
    def __init__(self, in_features, hid_features):
        super(RealNVP, self).__init__()
        out_features = in_features  # Flow is bijection!
        self.mask = None
        self.fc_linear = nn.Linear(in_features, hid_features)
        self.mean_linear = nn.Linear(hid_features, out_features)
        self.variance_linear = nn.Linear(hid_features, out_features)
        self.back_fc_linear = nn.Linear(in_features, hid_features)
        self.back_mean_linear = nn.Linear(hid_features, out_features)
        self.back_variance_linear = nn.Linear(hid_features, out_features)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self._mask = None
        self._mean = None
        self._std = None

    def forward_net(self, input, mask):
        activations = self.tanh(self.fc_linear(input * mask))
        mean = self.mean_linear(activations)
        std = self.sigmoid(self.variance_linear(activations))
        self._mean = mean
        self._std = std
        return mean, std

    def backward_net(self, input, mask):
        activations = self.tanh(self.back_fc_linear(input * mask))
        mean = self.back_mean_linear(activations)
        std = self.sigmoid(self.back_variance_linear(activations))
        return mean, std

    def forward(self, aux_curr: torch.Tensor, sample_mask=True):
        if sample_mask:
            self._mask = Bernoulli(.5).sample(aux_curr.shape)
        mask = self._mask
        mean, std = self.forward_net(aux_curr, mask)
        aux_next = mask * aux_curr + (1 - mask) * (aux_curr * std + (1 - std) * mean)
        log_det = torch.matmul((1 - mask).T, torch.log(std))
        return aux_next, log_det

    def pullback(self, aux_curr, sample_mask=False):
        if sample_mask:
            self._mask = Bernoulli(.5).sample(aux_curr.shape)

        mask = self._mask
        mean, std = self.backward_net(aux_curr, mask)

        aux_prev = mask * aux_curr + (1 - mask) * ((aux_curr - mean) / std - mean)
        log_det = - torch.matmul((1 - mask).T, torch.log(std))

        return aux_prev, log_det
