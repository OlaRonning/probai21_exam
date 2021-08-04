import math

import torch
from torch import nn
from torch.distributions import Normal, Bernoulli
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class Batch:
    def __init__(self, data):
        x, y = zip(*data)
        self.x = torch.stack(x)
        self.y = torch.stack(y)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self


class MNFG(nn.Module):  # TODO: rename
    def __init__(self, in_channels, hid_forward_flow, hid_inv_flow):
        super().__init__()
        bijections = (RealNVP(128, hid_forward_flow), RealNVP(128, hid_forward_flow))
        self.l1 = MNFLinear(in_channels, 128, bijections)  # reduction share the normalizing flow
        bijections = (RealNVP(32, hid_forward_flow), RealNVP(32, hid_forward_flow))
        self.l2 = MNFLinear(128, 32, bijections)
        bijections = (RealNVP(8, hid_forward_flow), RealNVP(8, hid_forward_flow))
        self.l3 = MNFLinear(32, 8, bijections)
        bijections = (RealNVP(1, hid_forward_flow), RealNVP(1, hid_forward_flow))
        self.l4 = MNFLinear(8, 1, bijections)
        self.layers = [self.l1, self.l2, self.l3, self.l4]

        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.l4(x)
        return x

    def log_likelihood(self, x, y):
        # assume gaussian likelihood
        return -.5 * torch.log(torch.tensor(2 * math.pi)) - .5 * (y - self(x)) ** 2

    def loss(self, x, y):
        kl = 0.
        ll = self.log_likelihood(x, y).sum()
        for l in self.layers: kl += l.kl_divergence()
        return ll + kl


class MultiplicativeNormalizingFlow:
    def __init__(self, in_channels, hid_forward_flow, hid_inv_flow, num_flows=2, num_epochs=100, optimizer=Adam,
                 batch_size=10):
        self.flow = MNFG(in_channels, hid_forward_flow, hid_inv_flow)
        self.num_epochs = num_epochs
        self.optimizer = optimizer(self.flow.parameters(), lr=1.)
        self.batch_size = batch_size

    def fit(self, xtr, ytr):
        loader = DataLoader(TensorDataset(xtr, ytr), batch_size=self.batch_size, collate_fn=lambda b: Batch(b),
                            pin_memory=True, shuffle=True)

        epoch_loss = float('inf')
        for _ in (t := trange(self.num_epochs)):
            cum_epoch_loss = 0.
            num_batchs = len(loader)
            for i, batch in enumerate(loader):
                self.optimizer.zero_grad()
                loss = self.flow.loss(batch.x, batch.y).mean()
                loss.backward()
                self.optimizer.step()
                cum_epoch_loss += loss
                t.set_description(f'Loss:{epoch_loss} ({i + 1}/{num_batchs})', )
            epoch_loss = cum_epoch_loss
            t.set_description(f'Loss:{epoch_loss} ({i + 1}/{num_batchs})', )

    def transform(self, xte):
        pass


class SubtractedMultiplicativeNormalizingFlow:
    def fit(self, xtr, ytr):
        pass

    def transform(self, xte):
        pass


class RealNVP(nn.Module):
    def __init__(self, in_features, hid_features):
        super(RealNVP, self).__init__()
        out_features = in_features  # Flow is bijection!
        self.mask = None
        self.fc_linear = nn.Linear(in_features, hid_features)
        self.mean_linear = nn.Linear(hid_features, out_features)
        self.variance_linear = nn.Linear(hid_features, out_features)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def net(self, input, mask):
        activations = self.tanh(self.fc_linear(input * mask))
        mean = self.mean_linear(activations)
        std = self.sigmoid(self.variance_linear(activations))
        return mean, std

    def forward(self, aux_curr: torch.Tensor):
        mask = Bernoulli(.5).sample(aux_curr.shape)
        self.mask = mask
        mean, std = self.net(aux_curr, mask)
        aux_next = mask * aux_curr + (1 - mask) * (aux_curr * std + (1 - std) * mean)
        log_det = torch.matmul((1 - mask).T, torch.log(std))
        return aux_next, log_det


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, forward_bijections, backward_bijections):
        super(MNFLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_func = nn.GELU()
        self.mean = torch.zeros((out_features, out_features), requires_grad=True)
        self.variance = torch.tensor(torch.eye(out_features), requires_grad=True)
        self.forward_bijections = forward_bijections
        self.backward_bijections = backward_bijections
        self.out_features = out_features
        self.in_features = in_features
        self.aux = None
        self.tanh = nn.Tanh()
        self.inv_dout = 1 / in_features * torch.ones((in_features))

    @property
    def base_dist(self):
        return Normal(0., 1.)

    @property
    def noise_dist(self):
        return Normal(0., 1.)

    def push_forward(self, aux0):
        self.aux0 = aux0
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
        return aux_back

    def forward(self, features):
        assert features.shape[-1] == self.in_features
        aux, _ = self.push_forward(aux0=self.base_dist.sample((self.out_features,)))
        activations = self.act_func(self.linear(features))
        latent_mean = torch.matmul(aux * activations, self.mean)
        latent_variance = torch.matmul(activations ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise

    def log_prob_aux_back(self, aux_back):
        loc = torch.outer(self.b1, self.tanh(self.c * self.linear.weight)) * self.inv_dout
        std = torch.outer(self.b2, self.tanh(self.c * self.linear.weight)) * self.inv_dout

        return torch.distributions.Normal(loc, std).log_prob(aux_back)

    def kl_divergence(self):
        kl_qw_pw = .5 * (self.variance + self.aux * self.mean + torch.log(self.variance))

        aux, log_det_aux = self.push_forward(self.aux0)
        log_prob_aux = self.base_dist.log_prob(self.aux0) - log_det_aux

        aux_back, log_det_aux_back = self.push_back(aux)
        log_prob_aux_back = self.log_prob_aux_back(aux_back) + log_det_aux_back

        return kl_qw_pw.sum() + log_prob_aux_back - log_prob_aux


class MNFConv(nn.Module):
    pass
