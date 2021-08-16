import argparse
from sys import stderr

from src.data_pipeline import load_data
from src.metrics import rmse, picp, mpiw
from src.models import MultiplicativeNormalizingFlow, NoisyNaturalGradient

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
        method = MultiplicativeNormalizingFlow(data.xtr.shape[-1], 50, 1, batch_size=1000, num_epochs=10_000)
    elif args.method == 4:
        method = NoisyNaturalGradient()
    else:
        raise KeyError(f'Method {args.method} not implemented.')

    print(args.dataset, file=stderr)
    method.fit(data.xtr, data.ytr)
    y_sim = method.transform(data.xte, 100)

    print(f'Method:{str(method)}, dataset:{args.dataset}, '
          f'RMSE:{rmse(data.yte, y_sim.mean(0)):.2f}, '
          f'PICP:{picp(data.yte, y_sim):.2f}, '
          f'MPIW:{mpiw(y_sim)}')
