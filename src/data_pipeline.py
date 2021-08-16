from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DATADIR = Path(__file__).parent.parent / 'datasets'
DataState = namedtuple('data', ['xtr', 'xte', 'ytr', 'yte'])


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f'{name}.txt')
    x, y = data[:, :-1], data[:, -1]
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=.60)
    return DataState(*map(partial(torch.tensor, dtype=torch.float), (xtr, xte, ytr, yte)))
