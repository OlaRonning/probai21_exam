from time import sleep

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


class MultiplicativeNormalizingFlow:
    def __init__(self, in_features, hid_features, out_features, num_epochs=4, optimizer=Adam, batch_size=10):
        bijections = (RealNVP(out_features, hid_features), RealNVP(out_features, hid_features))
        self.flow = MNFLinear(in_features, out_features, bijections)
        self.num_epochs = num_epochs
        self.optimizer = optimizer(self.flow.parameters())
        self.batch_size = batch_size

    def fit(self, xtr, ytr):
        loader = DataLoader(TensorDataset(xtr, ytr), batch_size=self.batch_size, collate_fn=lambda b: Batch(b),
                            pin_memory=True, shuffle=True)

        self.flow.forward(xtr)
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
        variance = self.sigmoid(self.variance_linear(activations))
        return mean, variance

    def forward(self, aux_curr: torch.Tensor):
        mask = Bernoulli(.5).sample(aux_curr.shape)
        self.mask = mask
        mean, variance = self.net(aux_curr, mask)
        aux_next = mask * aux_curr + (1 - mask) * (aux_curr * variance + (1 - variance) * mean)
        return aux_next


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, bijections):
        super(MNFLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_func = nn.GELU()
        self.mean = torch.zeros((out_features, out_features), requires_grad=True)
        self.variance = torch.tensor(torch.eye(out_features), requires_grad=True)
        self.bijections = bijections
        self.out_features = out_features
        self.in_features = in_features

    @property
    def base_dist(self):
        return Normal(0., 1.)

    @property
    def noise_dist(self):
        return Normal(0., 1.)

    def push_forward(self, aux0):
        aux = aux0
        for func in self.bijections:
            aux = func(aux)
        return aux

    def forward(self, features):
        assert features.shape[-1] == self.in_features
        aux = self.push_forward(aux0=self.base_dist.sample((self.out_features,)))
        activations = self.act_func(self.linear(features))
        latent_mean = torch.matmul(aux * activations, self.mean)
        latent_variance = torch.matmul(activations ** 2, self.variance)
        noise = self.noise_dist.sample(latent_variance.shape)
        return latent_mean + torch.sqrt(latent_variance) * noise

    def loss(self, x, y):
        sleep(.1)
        return torch.tensor(10., requires_grad=True)


class MNFConv(nn.Module):
    pass
