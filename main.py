import argparse
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from models import MultiplicativeNormalizingFlow
from noisy_natural_gradient import NoisyNaturalGradient

DATADIR = Path('datasets')
DataState = namedtuple('data', ['xtr', 'xte', 'ytr', 'yte'])


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f'{name}.txt')
    x, y = data[:, :-1], data[:, -1]
    return DataState(*map(partial(torch.tensor, dtype=torch.float), train_test_split(x, y, train_size=.60)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['boston_housing',
                                              'concrete',
                                              'energy_heating_load',
                                              'kin8nm',
                                              'naval_compressor_decay',
                                              'power',
                                              'protein',
                                              'wine',
                                              'yacht',
                                              'year_prediction_msd'], default='wine')
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]', default=2)

    args = parser.parse_args()
    data = load_data(args.dataset)

    if args.method == 0:
        raise NotImplementedError
    elif args.method == 2:
        method = MultiplicativeNormalizingFlow(data.xtr.shape[-1], 50, 1, batch_size=100)
    elif args.method == 4:
        method = NoisyNaturalGradient()
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    method.fit(data.xtr, data.ytr)

    ypred = method.transform(data.xte)
