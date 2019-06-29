import json
import numpy as np
import sys

from matplotlib import pyplot as plt
from itertools import product


def plot_new_loss_xp(filename):

    with open(filename, 'r') as data_file:
        data = json.load(data_file)['results']

    for j, (dataset, ds_size) in enumerate(product(['mnist', 'cifar'], ['normal', 'full'])):


        # get data pertaining to this set & size
        data_slice = list(filter(
            lambda d: (d['param_combination']['dataset'] == dataset
                       and d['param_combination']['dataset_size'] == ds_size),
            data
        ))

        # print experiment durations
        print([f"{d['duration']} - {d['param_combination']['loss_method']} - {dataset} {ds_size}" for d in data_slice])

        # plot it
        for i, d in product(range(len(data_slice) - 1), range(2)):
            plt.figure(i)
            plt.subplot(221+j)
            plt.title(f"{dataset} - {ds_size}")
            averages = np.mean(data_slice[i + d]['values'], 0)
            stds = np.std(data_slice[i + d]['values'], 0)

            plt.errorbar(
                range(len(averages)),
                averages,
                yerr=[2 * x for x in stds],
                label=data_slice[i + d]['param_combination']['loss_method'],
                capsize=5,
            )
            plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_new_loss_xp(sys.argv[1])
