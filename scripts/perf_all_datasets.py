from argparse import ArgumentParser
from pathlib import Path
from sys import stdout, stderr

from src.data_pipeline import load_data
from src.metrics import rmse, picp, mpiw
from src.mnf.multiplicative_normalizing_flow import MSFFeedForwardNetwork, MNFFeedForwardNetwork
from src.nng.noisy_adam import NoisyAdam
from src.svi import SVI


def print_row(i, table_length, dataset, method, yte, xte, file):
    y_sim = method.transform(xte, 100)
    rmse_ = rmse(yte, y_sim.mean(0))
    picp_ = picp(yte, y_sim)
    mpiw_ = mpiw(y_sim)
    method_col = ''
    if i == 0:
        method_col = '\\multirow{%d}{*}{%s}' % (table_length, str(method))

    print(method_col, dataset.replace('_', ' ').title(), round(rmse_, 2), round(picp_, 2), round(mpiw_, 2), sep=' & ',
          end=r'\\' + '\n', file=file)


def get_batch_size(dataset):
    batch_size = 1000
    if dataset == 'year_prediction_msd':
        batch_size = 70_000
    elif dataset == 'protein':
        batch_size = 10_000
    return batch_size


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=int, choices=range(5), metavar='[0-4]', default=4)
    parser.add_argument('--out_file', type=str, default='nng_perf.tex')

    args = parser.parse_args()

    datasets = [
        'year_prediction_msd',
        'boston_housing',
        'concrete',
        'energy_heating_load',
        'kin8nm',
        'naval_compressor_decay',
        'power',
        'protein',
        'wine',
        'yacht',
    ]
    if args.out_file == '':
        out_file = stdout
    else:
        table_dir = Path(__file__).parent.parent / 'out_tables'
        table_dir.mkdir(exist_ok=True)
        out_file = (table_dir / args.out_file).open('w')

    num_epochs = 2000

    method = lambda model, batch_size: SVI(model, batch_size=batch_size, num_epochs=num_epochs)
    if args.method == 0:
        bnn = lambda in_channels: MSFFeedForwardNetwork(in_channels, 50, 2)
    elif args.method == 2:
        bnn = lambda in_channels: MNFFeedForwardNetwork(in_channels, 50, 100, 2)
    elif args.method == 4:
        bnn = lambda _: None
        method = lambda _, batch_size: NoisyAdam(batch_size=batch_size, num_epochs=num_epochs)
    else:
        raise KeyError(f'Method {args.method} not implemented.')
    num_datasets = len(datasets)

    for i, dataset in enumerate(datasets):
        data = load_data(dataset)
        print(dataset, file=stderr)
        method_ = method(bnn(data.xtr.shape[-1]), get_batch_size(dataset))
        method_.fit(data.xtr, data.ytr)
        print_row(i, num_datasets, dataset, method_, data.yte, data.xte, out_file)
