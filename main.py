import argparse
from sys import stderr

from src.data_pipeline import load_data
from src.metrics import rmse, picp, mpiw
from src.mnf.multiplicative_normalizing_flow import MNFFeedForwardNetwork, MSFFeedForwardNetwork
from src.svi import SVI
from src.nng.bayes_ffn import BFeedForwardNetwork
from src.nng.optim import NoisyAdam

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

    args = parser.parse_args()
    data = load_data(args.dataset)

    in_channels = data.xtr.shape[-1]

    if args.method == 0:
        bnn = MSFFeedForwardNetwork(in_channels, 50, 2)
        method = SVI(bnn, batch_size=1000, num_epochs=1000)
    elif args.method == 2:
        bnn = MNFFeedForwardNetwork(in_channels, 50, 100, 2)
        method = SVI(bnn, batch_size=1000, num_epochs=1000)
    elif args.method == 4:
        bnn = BFeedForwardNetwork(in_channels)
        method = SVI(bnn, batch_size=1000, num_epochs=1000, optimizer=NoisyAdam)
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    print(args.dataset, file=stderr)
    method.fit(data.xtr, data.ytr)
    y_sim = method.transform(data.xte, 100)

    print(f'Method:{str(method)}, dataset:{args.dataset}, '
          f'RMSE:{rmse(data.yte, y_sim.mean(0)):.2f}, '
          f'PICP:{picp(data.yte, y_sim):.2f}, '
          f'MPIW:{mpiw(y_sim)}')
