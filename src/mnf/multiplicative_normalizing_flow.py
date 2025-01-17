import torch
from torch import nn

from src.mnf.flows import RealNVP
from src.mnf.layers import MNFLinear, MSFLinear


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

    def __repr__(self):
        return 'MNF'

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.l4(x)
        return x

    def log_likelihood(self, x, y):
        return torch.distributions.Normal(self(x).reshape(y.shape), .01).log_prob(y)

    def loss(self, x, y):
        return -(self.log_likelihood(x, y) + sum(layer.kl_divergence() for layer in self.layers)).mean()


class MSFFeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, hid_flow, num_flows):
        super().__init__()

        self.l1 = MSFLinear(in_channels, 128, build_flow(in_channels, hid_flow, num_flows))
        self.l2 = MSFLinear(128, 32, build_flow(128, hid_flow, num_flows))
        self.l3 = MSFLinear(32, 8, build_flow(32, hid_flow, num_flows))
        self.l4 = MSFLinear(8, 1, build_flow(8, hid_flow, num_flows))

        self.layers = [self.l1, self.l2, self.l3, self.l4]
        self.act = nn.Tanh()

    def __repr__(self):
        return 'MSF'

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.l4(x)
        return x

    def log_likelihood(self, x, y):
        return torch.distributions.Normal(self(x).reshape(y.shape), .01).log_prob(y)

    def loss(self, x, y):
        return -(self.log_likelihood(x, y) + sum(layer.kl_divergence() for layer in self.layers)).mean()


def build_flows(in_channels, hidden_forward_channels, hidden_backward_channels, num_flows,
                flow_constr=RealNVP):
    forward = build_flow(in_channels, hidden_forward_channels, num_flows, flow_constr)
    backward = build_flow(in_channels, hidden_backward_channels, num_flows, flow_constr)
    return forward, backward


def build_flow(in_channels, hidden_channels, num_flows, flow_constr=RealNVP):
    flow = nn.ModuleList(flow_constr(in_channels, hidden_channels) for _ in range(num_flows))
    return flow
