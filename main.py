import argparse
from collections import namedtuple
from pathlib import Path

from multiplicative_normalizing_flow import MultiplicativeNormalizingFlow, SubtractedMultiplicativeNormalizingFlow
from noisy_natural_gradient import NoisyNaturalGradient

DATADIR = Path('datasets')
DataState = namedtuple('data', ['xtr', 'ytr', 'xte', 'yte'])


def load_data(name: str) -> DataState:
    # TODO
    return DataState(None, None, None, None)


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
                                              'year_prediction_msd'])
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]')

    args = parser.parse_args()
    if args.method == 0:
        method = SubtractedMultiplicativeNormalizingFlow()
    elif args.method == 2:
        method = MultiplicativeNormalizingFlow()
    elif args.method == 4:
        method = NoisyNaturalGradient()
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    data = load_data(args.dataset)
    method.fit(data.xtr, data.ytr)

    ypred = method.transform(data.xte)
