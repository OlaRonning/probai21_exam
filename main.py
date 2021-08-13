import argparse
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.models import MultiplicativeNormalizingFlow, NoisyNaturalGradient

DATADIR = Path('datasets')
DataState = namedtuple('data', ['xtr', 'xte', 'ytr', 'yte'])


def load_data(name: str) -> DataState:
    data = np.loadtxt(DATADIR / f'{name}.txt')
    x, y = data[:, :-1], data[:, -1]
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=.60)
    xtr_mean = xtr.mean(0)
    xtr_std = np.std(xtr, 0)
    xtr = (xtr - xtr_mean) / xtr_std
    xte = (xte - xtr_mean) / xtr_std
    return DataState(*map(partial(torch.tensor, dtype=torch.float), (xtr, xte, ytr, yte)))


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
                                              'year_prediction_msd'], default='yacht')
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]', default=2)

    args = parser.parse_args()
    data = load_data(args.dataset)

    if args.method == 0:
        raise NotImplementedError
    elif args.method == 2:
        method = MultiplicativeNormalizingFlow(data.xtr.shape[-1], 50, 1, batch_size=100, num_epochs=1000)
    elif args.method == 4:
        method = NoisyNaturalGradient()
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    method.fit(data.xtr, data.ytr)

    ypred = method.transform(data.xte)
