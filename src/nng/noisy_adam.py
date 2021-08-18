import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import matplotlib.pyplot as plt

from src.svi import Batch
from torch.autograd import grad


def bnn(x, weights, shapes):
    y = x
    weights = torch.split(weights, tuple(s.numel() for s in shapes))
    weights, biases = weights[::2], weights[1::2]
    for weight, bias, shape in zip(weights, biases, shapes[::2]):
        y = torch.tanh(torch.matmul(y, weight.reshape(shape)) + bias)
    return y.squeeze()


def log_likelihood(x, y, weights, shapes):
    return torch.distributions.Normal(bnn(x, weights, shapes), .01).log_prob(y).mean()


class NoisyAdam:
    def __init__(self, num_epochs, lr=.01, beta1=.9, beta2=.999, kl_weight=1., pvar=.1, damping_ex=.1, verbose=True,
                 batch_size=1000):
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.kl_weight = kl_weight
        self.pvar = pvar
        self.damping_ex = damping_ex
        self.verbose = verbose
        self.debug = False
        self.batch_size = batch_size
        self._mus = None
        self._fs = None
        self._shapes = None
        self._n = None
        self._damping_in = None

    def __repr__(self):
        return 'NNG'

    def fit(self, xtr, ytr):
        n, m = xtr.shape
        loader = DataLoader(TensorDataset(xtr, ytr), batch_size=self.batch_size, collate_fn=lambda b: Batch(b),
                            pin_memory=True, shuffle=True)
        losses = []
        damping_in = self.kl_weight / (n * self.pvar)
        damping = self.damping_ex + damping_in

        shapes = (
            torch.Size((m, 132)), torch.Size((132,)), torch.Size((132, 32)), torch.Size((32,)), torch.Size((32, 8)),
            torch.Size((8,)), torch.Size((8, 1)), torch.Size((1,)))
        self._shapes = shapes
        self._n = n
        mus = torch.cat(tuple(torch.ones(s.numel()) for s in shapes))
        fs = .001 * torch.cat(tuple(torch.ones(s.numel()) for s in shapes))
        m = 0.
        epoch_loss = float('inf')
        for k in (t := trange(self.num_epochs) if self.verbose else range(self.num_epochs)):
            cum_epoch_loss = 0.
            num_batchs = len(loader)
            for i, batch in enumerate(loader):
                weights = self._sample_w(mus, fs, damping_in)
                weights.requires_grad = True
                loss = log_likelihood(batch.x, batch.y, weights, shapes)
                cum_epoch_loss += loss
                dloss_w = torch.cat(grad(loss, weights))

                v = dloss_w - damping_in * weights
                m = self.beta1 * m + (1 - self.beta1) * v
                fs = self.beta2 * fs + (1 - self.beta2) * dloss_w ** 2
                mus += self.lr * m / ((1 - self.beta1 ** (k + 1)) * (fs + damping))

                if self.verbose:
                    t.set_description(f'Loss:{epoch_loss:.2f} ({i + 1}/{num_batchs})', )
            epoch_loss = cum_epoch_loss
            losses.append(epoch_loss.detach().numpy())
            if self.verbose:
                t.set_description(f'Loss:{epoch_loss:.2f} ({i + 1}/{num_batchs})', )
        self._mus = mus
        self._fs = fs
        self._damping_in = damping_in

        if self.debug:
            plt.plot(losses)
            plt.show()

    def _sample_w(self, mus, fs, damping_in):
        return torch.distributions.Normal(mus, self.kl_weight / (self._n) * 1 / (fs + damping_in)).sample(())

    def transform(self, x, num_samples=1):
        return torch.stack(tuple(bnn(x, self._sample_w(self._mus, self._fs, self._damping_in), self._shapes) for _ in
                                 range(num_samples))).squeeze()
