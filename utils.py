import torch
from torch import nn

from flows import RealNVP


class Batch:
    def __init__(self, data):
        x, y = zip(*data)
        self.x = torch.stack(x)
        self.y = torch.stack(y)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self


def build_flows(in_channels, hidden_forward_channels, hidden_backward_channels, num_flows,
                flow_constr=RealNVP):
    forward = nn.ModuleList(flow_constr(in_channels, hidden_forward_channels) for _ in range(num_flows))
    backward = nn.ModuleList(flow_constr(in_channels, hidden_backward_channels) for _ in range(num_flows))
    return forward, backward
