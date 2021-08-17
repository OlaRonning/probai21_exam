import argparse
from sys import stderr

from src.data_pipeline import load_data
from src.metrics import rmse, picp, mpiw
from src.mnf.multiplicative_normalizing_flow import MNFFeedForwardNetwork, MSFFeedForwardNetwork
from src.svi import SVI
from src.nng.bayes_ffn import BFeedForwardNetwork
from src.nng.optim import NoisyAdam


def get_batch_size(dataset):
    batch_size = 1000
    if dataset == 'year_prediction_msd':
        batch_size = 70_000
    elif dataset == 'protein':
        batch_size = 10_000
    return batch_size


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
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]', default=0)
    parser.add_argument('--verbose', type=bool, default=True)

    args = parser.parse_args()
    data = load_data(args.dataset)

    in_channels = data.xtr.shape[-1]

    batch_size = get_batch_size(args.dataset)
    num_epochs = 2000

    if args.method == 0:
        bnn = MSFFeedForwardNetwork(in_channels, 50, 2)
        method = SVI(bnn, batch_size=batch_size, num_epochs=num_epochs, verbose=args.verbose)
    elif args.method == 2:
        bnn = MNFFeedForwardNetwork(in_channels, 50, 100, 2)
        method = SVI(bnn, batch_size=batch_size, num_epochs=num_epochs, verbose=args.verbose)
    elif args.method == 4:
        bnn = BFeedForwardNetwork(in_channels)
        method = SVI(bnn, batch_size=batch_size, num_epochs=num_epochs, optimizer=NoisyAdam, verbose=args.verbose)
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    if args.verbose:
        print(args.dataset, file=stderr)
    method.fit(data.xtr, data.ytr)
    y_sim = method.transform(data.xte, 100)

    print(f'Method:{str(method)}({args.method}), dataset:{args.dataset}, '
          f'RMSE:{rmse(data.yte, y_sim.mean(0)):.2f}, '
          f'PICP:{picp(data.yte, y_sim):.2f}, '
          f'MPIW:{mpiw(y_sim)}')
