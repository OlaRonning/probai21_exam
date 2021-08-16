import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from src.mnf.multiplicative_normalizing_flow import MNFFeedForwardNetwork


class MultiplicativeNormalizingFlow:
    def __init__(self, in_channels, hid_forward_flow, hid_inv_flow, num_flows=2, num_epochs=1000, optimizer=Adam,
                 batch_size=100):
        self.bnn = MNFFeedForwardNetwork(in_channels, hid_forward_flow, hid_inv_flow, num_flows)
        self.num_epochs = num_epochs
        self.optimizer = optimizer(self.bnn.parameters())
        self.batch_size = batch_size
        self.debug = False

    def __repr__(self):
        return 'MNF'

    def fit(self, xtr, ytr):
        loader = DataLoader(TensorDataset(xtr, ytr), batch_size=self.batch_size, collate_fn=lambda b: Batch(b),
                            pin_memory=True, shuffle=True)

        epoch_loss = float('inf')
        losses = []
        for _ in (t := trange(self.num_epochs)):
            cum_epoch_loss = 0.
            num_batchs = len(loader)
            for i, batch in enumerate(loader):
                self.optimizer.zero_grad()
                loss = self.bnn.loss(batch.x, batch.y)
                loss.backward()
                self.optimizer.step()
                cum_epoch_loss += loss
                t.set_description(f'Loss:{epoch_loss:.2f} ({i + 1}/{num_batchs})', )
            epoch_loss = cum_epoch_loss
            losses.append(epoch_loss.detach().numpy())
            t.set_description(f'Loss:{epoch_loss:.2f} ({i + 1}/{num_batchs})', )
        if self.debug:
            plt.plot(losses)
            plt.show()

    def transform(self, x, num_samples=1):
        return torch.stack(tuple(self.bnn(x) for _ in range(num_samples))).squeeze()


class NoisyNaturalGradient():
    def fit(self, xtr, ytr):
        pass

    def transform(self, xte):
        pass


class Batch:
    def __init__(self, data):
        x, y = zip(*data)
        self.x = torch.stack(x)
        self.y = torch.stack(y)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self
