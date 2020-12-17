from abc import ABC, abstractmethod
from random import choice as random_choice
from typing import List, Dict, Tuple, Generator, Set, Union

import numpy as np


class Distribution(ABC):
    """
    Abstract base class for distributions
    """

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        This method must be called before any other method. It performs necessary internal preprocessing.

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_value_possible(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_random_value(self, *args, **kwargs):
        pass


class DiscreteDistribution(Distribution):
    """
    This class represents discrete distribution.
    """

    def __init__(self, distribution: Dict):
        """
        Constructor for DiscreteDistribution class

        :param distribution: A dictionary with defined possible variables as keys,
         and probabilities as values e.g. {'A': 0.1, 'B': 0.9}
        :type distribution: Dict[str, float]
        """
        assert isinstance(distribution, dict) and all(
            isinstance(key, str) and isinstance(value, float) for key, value in distribution.items()) and sum(
            [x for x in distribution.values()]) == 1
        self.distribution = distribution
        self._values = None
        self._weights = None
        self._is_preprocessed = False

    def preprocess(self) -> None:
        """
        This method must be called before any other method. It performs necessary internal preprocessing.

        :return: None
        :rtype: None
        """
        assert self.distribution is not None

        self._values = list(self.distribution.keys())
        self._weights = np.array([self.distribution[key] for key in self._values], dtype=np.float32)

        self._is_preprocessed = True

    def sample(self, num_of_samples: int = 1) -> Union[str, List[str]]:
        """
        Return sample(s) from distribution
        :param num_of_samples: Number of samples to be returned
        :type num_of_samples: int
        :return: A single sample or a list of samples
        :rtype: Union[str, List[str]]
        """
        assert self._is_preprocessed, 'Distribution first must be preprocessed'
        samples = np.random.choice(self._values, num_of_samples, p=self._weights)
        if num_of_samples == 1:
            return samples[0]
        return list(samples)

    def is_value_possible(self, value: str) -> bool:
        """
        Returns if value is possible in distribution
        :param value: Value to check
        :type value: str
        :return: True if value is in distribution or False otherwise
        :rtype: bool
        """
        assert self._is_preprocessed, 'Distribution first must be preprocessed'
        return value in self._values

    def get_random_value(self):
        """
        Return a random possible value, but with uniform distribution

        :return: Random possible in distribution value drawn with uniform distribution
        :rtype: str
        """
        assert self._values, 'Distribution first must be preprocessed'
        return random_choice(self._values)


class ConditionalDistribution(Distribution):
    """
    This class represents conditional distribution.
    """

    def __init__(self, distribution: List[List[Union[str, float]]]):
        """
        Constructor for DiscreteDistribution class

        :param distribution: List of lists describing conditional probabilities. Last value in list is a probability,
        and second last is a value in that distribution. Values at positions 0:-2 are evidences.

        Example:
        [
            ['A', 'X', 0.5],
            ['A', 'Y', 0.5],
            ['B', 'X', 0.3],
            ['B', 'Y', 0.7]
        ]

        Possible values in this example distribution are X and Y
        :type distribution: List[List[Union[str,float]]]
        """
        assert isinstance(distribution, list) and all(isinstance(x, list) for x in distribution)
        for x in distribution:
            assert all(isinstance(y, str) for y in x[:-1]) and (isinstance(x[-1], float) or (isinstance(x[-1], int)))
        self.distribution = distribution
        self.dist_len = len(self.distribution)

        # last two values in a row in table represents value, and probability given evidence
        self.num_of_dependencies = len(distribution[0]) - 2
        self.conditional_distribution_lookup = dict()
        self._is_preprocessed = False
        self._values = None

    def preprocess(self):
        """
        This method must be called before any other method. It performs necessary internal preprocessing.

        :return: None
        :rtype: None
        """
        assert self.distribution is not None

        def get_possible_values_and_weight_for_evidence(distribution, evidence) -> Tuple[List[str], np.array]:
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
        # create a lookup dictionary for probabilities given evidence
        for possible_evidence in possible_evidences:
            self.conditional_distribution_lookup[possible_evidence] = get_possible_values_and_weight_for_evidence(
                self.distribution, possible_evidence)

        self._is_preprocessed = True

    def sample(self, evidence: List[str], num_of_samples: int = 1) -> Union[str, List[str]]:
        """
        Return sample(s) from distribution given evidence
        :param num_of_samples: Number of samples to be returned
        :type num_of_samples: int
        :return: A single sample or a list of samples
        :rtype: Union[str, List[str]]
        """
        assert self._is_preprocessed, 'Distribution first must be preprocessed'

        evidence = tuple(evidence)
        samples = np.random.choice(
            self.conditional_distribution_lookup[evidence][0], num_of_samples,
            p=self.conditional_distribution_lookup[evidence][1])[:][0]
        if num_of_samples == 1:
            return samples
        return samples

    def is_value_possible(self, value: str):
        """
        Returns if value is possible in distribution

        :param value: Value to check
        :type value: str
        :return: True if value is in distribution or False otherwise
        :rtype: bool
        """
        return value in self._values

    def get_random_value(self):
        """
        Return a random possible value, but with uniform distribution

        :return: Random possible in distribution value drawn with uniform distribution
        :rtype: str
        """
        assert self._values, 'To get random value, distribution first must be preprocessed'
        return random_choice(self._values)

    def get_dependencies_possible_values(self) -> Generator[Set[str], None, None]:
        """
        Generator that returns possible possible values for i-th dependency

        :return: Generator for possible values in i-th column in conditional probability table (except last two columns)
        :rtype: Generator[Set[str]]
        """
        for i in range(self.num_of_dependencies - 2):
            ith_column = [x[i] for x in self.distribution]
            possible_values = set(ith_column)

            yield possible_values
