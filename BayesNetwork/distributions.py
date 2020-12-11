from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np


class Distribution(ABC):

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_value_possible(self, *args, **kwargs):
        pass


class DiscreteDistribution(Distribution):

    def __init__(self, distribution: Dict):
        # TODO change to exceptions
        assert isinstance(distribution, dict) and all(
            isinstance(key, str) and isinstance(value, float) for key, value in distribution.items()) and sum(
            [x for x in distribution.values()]) == 1
        self.distribution = distribution
        self._values = None
        self._weights = None
        self._is_preprocessed = False

    def preprocess(self):
        assert self.distribution is not None

        self._values = list(self.distribution.keys())
        self._weights = np.array([self.distribution[key] for key in self._values], dtype=np.float32)

        self._is_preprocessed = True

    def sample(self, num_of_samples: int = 1):
        assert self._is_preprocessed, 'Distribution first must be preprocessed'
        samples = np.random.choice(self._values, num_of_samples, p=self._weights)
        if num_of_samples == 1:
            return samples[0]
        return samples

    def is_value_possible(self, value):
        return value in self._values


class ConditionalDistribution(Distribution):

    def __init__(self, distribution: List[List]):
        assert isinstance(distribution, list) and all(isinstance(x, list) for x in distribution)
        for x in distribution:
            assert all(isinstance(y, str) for y in x[:-1]) and isinstance(x[-1], float)
        self.distribution = distribution
        self.dist_len = len(self.distribution)

        # last two values in a row in table represents value, and probability given evidence
        self.num_of_dependencies = len(distribution[0]) - 2
        self.conditional_distribution_lookup = dict()
        self._is_preprocessed = False
        self._values = None

    def preprocess(self):
        assert self.distribution is not None

        def get_possible_values_and_weight_for_evidence(distribution, evidence) -> Tuple:
            values = []
            weights = []
            for x in distribution:
                if tuple(x[:self.num_of_dependencies]) == evidence:
                    values.append(x[-2])
                    weights.append(x[-1])

            weights = np.array(weights)
            assert sum(weights) == 1
            return values, weights

        self._values = list(set([x[-2] for x in self.distribution]))
        possible_evidences = list(set([tuple(x[:self.num_of_dependencies]) for x in self.distribution]))
        for possible_evidence in possible_evidences:
            self.conditional_distribution_lookup[possible_evidence] = get_possible_values_and_weight_for_evidence(
                self.distribution, possible_evidence)

        self._is_preprocessed = True

    def sample(self, evidence: List[str], num_of_samples: int = 1):
        assert self._is_preprocessed, 'Distribution first must be preprocessed'

        evidence = tuple(evidence)
        samples = np.random.choice(
            self.conditional_distribution_lookup[evidence][0], num_of_samples,
            p=self.conditional_distribution_lookup[evidence][1])[:][0]
        if num_of_samples == 1:
            return samples[0]
        return samples

    def is_value_possible(self, value):
        return value in self._values
