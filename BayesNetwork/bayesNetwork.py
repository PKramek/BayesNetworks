from typing import List, Dict

from BayesNetwork.distributions import Distribution, ConditionalDistribution


class Node:
    def __init__(self, distribution, name: str):
        assert isinstance(name, str), 'Name is not a string'
        assert isinstance(distribution, Distribution), 'Distribution must be of class Distribution'
        self.children = {}
        self.parents = {}
        self.markov_neighborhood = {}
        self.distribution = distribution
        self.name = name  # critical !!

        self.is_dependent = isinstance(distribution, ConditionalDistribution)
        self.counter = {}
        self.evidence = None

    def add_parent(self, parent):
        assert isinstance(parent, Node), 'Given parent is not a Node'
        if parent.name in self.children.keys():
            raise ValueError('Can not add parent, because not is already a child')
        if parent.name not in self.parents.keys():
            self.parents[parent.name] = parent
            self.markov_neighborhood[parent.name] = parent
        else:
            raise ValueError('Parent already know')

    def add_child(self, child):
        assert isinstance(child, Node)
        if child.name in self.parents.keys():
            raise ValueError('Can not add child, because not is already a parent')
        if child.name not in self.children.keys():
            self.children[child.name] = child
            self.markov_neighborhood[child.name] = child

    def get_markov_neighborhood(self) -> Dict:
        return self.markov_neighborhood

    def get_children(self) -> Dict:
        return self.children

    def get_parents(self) -> Dict:
        return self.parents

    def preprocess(self):
        self.distribution.preprocess()

    def reset_counters(self):
        self.counter = {}

    def sample(self, evidence=None):
        if self.evidence is None:
            if self.is_dependent:
                assert evidence is not None, 'Evidence must be given if node is dependent'
                sample = self.distribution.sample(evidence)
            else:
                sample = self.distribution.sample()
            self.counter.setdefault(sample, 0)
            self.counter[sample] += 1
        else:
            sample = self.evidence

        return sample

    def set_evidence(self, value):
        assert self.distribution.is_value_possible(value), 'Value not found in distribution'
        self.evidence = value

    def set_random(self):
        self.evidence = None


class BayesNetwork:
    def __init__(self):
        self.states = {}
        self.edges = []

    def add_states(self, list_of_states: List[Node]):
        assert isinstance(list_of_states, List) and all(isinstance(x, Node) for x in list_of_states)
        for state in list_of_states:
            if state.name not in self.states.keys():
                self.states[state.name] = state
            else:
                raise ValueError('States names must be unique')

    def add_edge(self, parent: Node, child: Node):
        assert isinstance(parent, Node) and isinstance(child, Node)

        if parent.name not in self.states.keys():
            raise ValueError('Not known parent node {}, it first must be added to list of stated'.format(parent))

        if child.name not in self.states.keys():
            raise ValueError('Not known child node {}, it first must be added to list of stated'.format(child))

        # TODO check if the order is right
        parent.add_child(child)
        child.add_parent(parent)

    def preprocess(self):
        assert self.states, 'No nodes in graph'
        for node in self.states.values():
            node.preprocess()

    def mcmc(self, num_of_repetitions: int):
        pass
