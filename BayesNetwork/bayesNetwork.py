from random import choice as random_choice
from typing import List, Dict

from BayesNetwork.distributions import Distribution, ConditionalDistribution


class Node:
    def __init__(self, distribution: 'Distribution', name: str):
        assert isinstance(name, str), 'Name is not a string'
        assert isinstance(distribution, Distribution), 'Distribution must be of class Distribution'
        self.children = {}
        self.parents = {}
        # Used in conditional probability
        self.parents_order = []

        self.markov_blanket = {}
        self.distribution = distribution
        self.name = name

        self.is_dependent = isinstance(distribution, ConditionalDistribution)
        self.counter = {}
        self.evidence = None
        self.static_value = None

    def add_parent(self, parent: 'Node'):
        assert isinstance(parent, Node), 'Given parent is not a Node'
        assert parent is not self, 'Cant add self as a parent'
        assert isinstance(self.distribution, ConditionalDistribution), 'Cant add parents to independent distribution'

        if parent.name in self.children.keys():
            raise ValueError('Can`t add parent, because not is already a child')
        if parent.name not in self.parents.keys():
            self.parents[parent.name] = parent
            self.parents_order.append(parent.name)
            self.markov_blanket[parent.name] = parent
        else:
            raise ValueError('Parent already know')

    def add_child(self, child: 'Node'):
        assert isinstance(child, Node)
        assert child is not self, 'Cant add self as a child'
        if child.name in self.parents.keys():
            raise ValueError('Can not add child, because not is already a parent')
        if child.name not in self.children.keys():
            self.children[child.name] = child
            self.markov_blanket[child.name] = child

    def get_markov_blanket(self) -> Dict:
        return self.markov_blanket

    def get_children(self) -> Dict:
        return self.children

    def get_parents(self) -> Dict:
        return self.parents

    def preprocess(self):
        # Adding parents children to markov blanket (excluding node itself)
        for child in self.children.values():
            for name, node in child.get_parents().items():
                if name != self.name:
                    self.markov_blanket[name] = node

        self.distribution.preprocess()

        if isinstance(self.distribution, ConditionalDistribution):
            # Check if parents are in the same order as dependencies in distribution table
            for possible_values in self.distribution.get_dependencies_possible_values():
                for i, value in enumerate(possible_values):
                    if not self.parents[self.parents_order[i]].is_value_possible():
                        raise RuntimeError('Parents for node: {} are out of order with given distribution'.format(
                            self.name
                        ))

    def reset_counters(self):
        self.counter = {}

    def sample(self, observations: List[str] = None):
        if self.evidence is None and self.static_value is None:
            if self.is_dependent:
                assert observations is not None, 'Observations must be given if node is dependent'
                sample = self.distribution.sample(observations)
            else:
                sample = self.distribution.sample()
            self.counter.setdefault(sample, 0)
            self.counter[sample] += 1
        elif self.evidence is not None:
            sample = self.evidence
        else:
            sample = self.static_value

        return sample

    def sample_given_markov_blanket(self, markov_blanket: Dict['str', 'Node']):
        observations = []
        for name in self.parents_order:
            observations.append(markov_blanket[name].sample())

        return self.sample(observations)

    def set_evidence(self, value: str):
        assert self.distribution.is_value_possible(value), 'Value not found in distribution'
        self.evidence = value

    def set_static_value(self, value: str):
        assert self.distribution.is_value_possible(value), 'Value not found in distribution'
        self.static_value = value

    def set_non_static(self):
        self.static_value = None

    def set_random_initial_value(self):
        self.static_value = self.distribution.get_random_value()

    def get_prob(self):
        if self.counter:
            total_occurrences = sum(self.counter.values())
            return {key: value / float(total_occurrences) for key, value in self.counter.items()}
        else:
            return None

    def is_value_possible(self, value: str):
        return self.distribution.is_value_possible(value)


class BayesNetwork:
    def __init__(self):
        self.states = {}

    def add_states(self, list_of_states: List['Node']):
        assert isinstance(list_of_states, List) and all(isinstance(x, Node) for x in list_of_states)
        for state in list_of_states:
            if state.name not in self.states.keys():
                self.states[state.name] = state
            else:
                raise ValueError('States names must be unique')

    def add_edge(self, parent: 'Node', child: 'Node'):
        assert isinstance(parent, Node) and isinstance(child, Node)

        if parent.name not in self.states.keys():
            raise ValueError('Not known parent node {}, it first must be added to list of stated'.format(parent))

        if child.name not in self.states.keys():
            raise ValueError('Not known child node {}, it first must be added to list of stated'.format(child))

        parent.add_child(child)
        child.add_parent(parent)

    def preprocess(self):
        assert self.states, 'No nodes in graph'
        for node in self.states.values():
            node.preprocess()

    def _set_evidences(self, evidence: Dict[str, str]):
        for name, state in evidence.items():
            self.states[name].set_evidence(state)

    def _check_query(self, query):
        for node in query:
            assert node in self.states.values(), 'Node {} not known'.format(node.name)

    def _reset_counters(self):
        assert self.states
        for state in self.states.values():
            state.reset_counters()

    def _get_states_without_evidence(self, evidence: Dict[str, str]):
        states_without_evidence = []
        for state in self.states.values():
            if state.name not in evidence.keys():
                states_without_evidence.append(state)

        return states_without_evidence

    def gibbs(self, evidence: Dict[str, str], query: List[Node], n: int):
        assert self.states, 'No nodes added to network'

        for node in query:
            if node.name in evidence.keys():
                raise ValueError('Node {} is in query as well as in evidence'.format(node.name))

        self._check_query(query)
        self._set_evidences(evidence)
        self._reset_counters()

        states_without_evidence = self._get_states_without_evidence(evidence)
        if not states_without_evidence:
            raise ValueError('Every node was given evidence, cant generate any data')

        # setting random initial values
        for node in states_without_evidence:
            node.set_random_initial_value()

        for i in range(n):
            node = random_choice(states_without_evidence)
            markov_blanket = node.get_markov_blanket()

            node.set_non_static()
            value = node.sample_given_markov_blanket(markov_blanket)
            node.set_static_value(value)

        results = {}
        for node in query:
            results[node.name] = node.get_prob()

        return results
