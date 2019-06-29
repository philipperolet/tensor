# coding: utf-8
import itertools
import numpy as np
import logging
import time
import json

from pprint import pprint


class Experimenter(object):

    def __init__(self, method, step_display=None, verbose=False):
        """
        method - method on which to run experiments
        step_display - method to display detailed result each time a param comb is tested
        """
        self.method = method
        self.step_display = step_display
        self.verbose = verbose

    def experiment(self, parameters_dict, iterations, json_dump=False, name=None):
        """
        parameters_dict: a dict of parameters, with each a list of values to try,
        iterations: the number of times the test should be repeated
        json_dump: if True, saves the result as json in a file
        Output: a list of dicts {method, param_combination: {dict}, avg, std, values}
        """
        logging.info(f"Experimenting {self.method.__name__} with {iterations}")

        results = {
            "method_name": self.method.__name__,
            "iterations": iterations,
            "start_time": time.time(),
            "results": [],
        }

        for params in itertools.product(*parameters_dict.values()):
            start_time = time.time()
            result = self._experiment_on_params(
                self._get_params_list(params, parameters_dict),
                iterations
            )
            result['duration'] = time.time() - start_time
            results["results"].append(result)

            if self.step_display is not None:
                self.step_display(result)

        if json_dump:
            self._save_as_json(results, name)

        results['duration'] = time.time() - results['start_time']
        return results

    def _save_as_json(self, results, name=None):
        filename = f"results/{self.method.__name__}_{int(time.time())}.json"
        if name:
            filename = f"experiments/{name}.{time.time()}.result.json"
        with open(filename, 'w') as res_file:
            json.dump(results, res_file, default=str)

    def _experiment_on_params(self, param_combination, iterations):
        params_result = {
            "values": [],
            "param_combination": param_combination,
        }

        for step in range(iterations):
            params_result['values'].append(self.method(**param_combination))
            if self.verbose:
                print("Iteration {}, result: {}".format(step, params_result['values'][-1]))

        params_result['avg'] = np.mean(params_result['values'], 0)
        params_result['std'] = np.std(params_result['values'], 0)

        if self.verbose:
            pprint(params_result)

        return params_result

    def _get_params_list(self, params, parameters_dict):
        param_combination = dict()
        for i, k in enumerate(parameters_dict.keys()):
            param_combination[k] = params[i]
        return param_combination


def get_data_from_json(filename):
    with open(filename, 'r') as data_file:
        return json.load(data_file)
