# coding: utf-8
import itertools
import numpy as np
from pprint import pprint


class Experimenter(object):

    def __init__(self, method, verbose=True):
        self.method = method
        self.verbose = verbose

    def experiment(self, parameters_dict, iterations):
        """
        Input: a method on which to run experiments, a dict of parameters, with each a list of
        values to try, the number of times the test should be repeated
        Output: a list of dicts {method: , param_combination: {dict}, avg, std, values}
        """
        results = []

        for params in itertools.product(*parameters_dict.values()):
            result = self._experiment_on_params(
                self._get_params_list(params, parameters_dict),
                iterations
            )
            results.append(result)

        return results

    def _experiment_on_params(self, param_combination, iterations):
        params_result = {
            "method": self.method.__name__,
            "values": [],
            "iterations": iterations,
            "param_combination": param_combination,
        }
        if self.verbose:
            pprint(params_result)

        for step in range(iterations):
            params_result['values'].append(self.method(**param_combination))
            if self.verbose:
                print("Iteration {}, result: {}".format(step, params_result['values']))

        params_result['avg'] = np.mean(params_result['values'])
        params_result['std'] = np.std(params_result['values'])
        return params_result

    def _get_params_list(self, params, parameters_dict):
        param_combination = dict()
        for i, k in enumerate(parameters_dict.keys()):
            param_combination[k] = params[i]
        return param_combination
