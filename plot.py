from matplotlib import pyplot as plt
import json
from itertools import product
import sys


def plot_with_error_bars(filename):

    with open(filename, 'r') as data_file:
        data = json.load(data_file)['results']

    # turn exp_factor None to 0
    for d in data:
        if not(d['param_combination']['exp_factor']):
            d['param_combination']['exp_factor'] = 0

    for i, (dataset, ds_size) in enumerate(product(['mnist', 'cifar'], ['normal', 'full'])):
        plt.subplot(221 + i)
        plt.title(f"{dataset} - {ds_size}")

        # get data pertaining to this set & size
        data_slice = filter(
            lambda d: (d['param_combination']['dataset'] == dataset
                       and d['param_combination']['dataset_size'] == ds_size),
            data
        )

        # sort it according to exp_factor
        data_slice = sorted(
            data_slice,
            key=lambda d: d['param_combination']['exp_factor']
        )

        # plot it
        plt.errorbar(
            [d['param_combination']['exp_factor'] for d in data_slice],
            [d['avg'] for d in data_slice],
            yerr=[d['std'] * 2 for d in data_slice]
        )
    plt.show()


if __name__ == "__main__":
    plot_with_error_bars(sys.argv[1])
