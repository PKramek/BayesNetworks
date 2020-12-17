from random import choice as random_choice
from typing import List, Dict, Union

from BayesNetwork.distributions import Distribution, ConditionalDistribution


class Node:
    """
    Class representing node in Bayes network
    """

    def __init__(self, distribution: 'Distribution', name: str):
        """
        Constructor for Node class

        :param distribution: Distribution class object
        :type distribution: 'Distribution'
        :param name: Name of the node
        :type name: str
        """
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

    def add_parent(self, parent: 'Node') -> None:
        """
        Add parent to node

        :param parent: Parent Node
        :type parent: Node
        :return: None
        :rtype: None
        """
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
        """
        Add child to node

        :param child: Child Node
        :type child: Node
        :return: None
        :rtype: None
        """
        assert isinstance(child, Node)
        assert child is not self, 'Cant add self as a child'
        if child.name in self.parents.keys():
            raise ValueError('Can not add child, because not is already a parent')
        if child.name not in self.children.keys():
            self.children[child.name] = child
            self.markov_blanket[child.name] = child

    def get_markov_blanket(self) -> Dict[str, 'Node']:
        """
        Return Markov Blanket for Node

        :return: Markov Blanket for Node
        :rtype: Dict[str, 'Node']
        """
        return self.markov_blanket

    def get_children(self) -> Dict[str, 'Node']:
        """
        Return children of Node

        :return: Node`s children
        :rtype: Dict[str, 'Node']
        """
        return self.children

    def get_parents(self) -> Dict[str, 'Node']:
        """
        Return parents of Node

        :return: Node`s parents
        :rtype: Dict[str, 'Node']
        """
        return self.parents

    def preprocess(self):
        """
        This method must be called before any other method. It performs necessary internal preprocessing.

        :return: None
        :rtype: None
        """
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
        """
        Reset occurrence counters
        :return:
        :rtype:
        """
        self.counter = {}

    def sample(self, observations: List[str] = None) -> str:
        """
        Sample from Node distribution. If either evidence is set or value is set to static value is not drawn
        from distribution but either evidence or static value is returned
        :param observations: List of observations
        :type observations: List[str]
        :return: Sample from node distribution
        :rtype: str
        """
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

    def sample_given_markov_blanket(self, markov_blanket: Dict[str, 'Node']):
        """
        Returns a sample given markov blanket.

        :param markov_blanket: Nodes markov blanket
        :type markov_blanket: Dict[str, 'Node']
        :return: Sample from nodes distribution
        :rtype: str
        """
        observations = []
        for name in self.parents_order:
            observations.append(markov_blanket[name].sample())

        return self.sample(observations)

    def set_evidence(self, value: str) -> None:
        """
        Set evidence in node. When evidence is set, sample will return its value

        :param value: Value to be set as evidence
        :type value: str
        :return: None
        :rtype: None
        """
        assert self.distribution.is_value_possible(value), 'Value not found in distribution'
        self.evidence = value

    def set_static_value(self, value: str) -> None:
        """
        Set static value in node. When static value is set, sample will return its value

        :param value: Value to be set as evidence
        :type value: str
        :return: None
        :rtype: None
        """
        assert self.distribution.is_value_possible(value), 'Value not found in distribution'
        self.static_value = value

    def set_non_static(self):
        """
        Sets node as non static

        :return: None
        :rtype: Node
        """
        self.static_value = None

    def set_random_initial_value(self):
        """
        Sets nodes static value as random sample drawn with uniform distribution from possible values in distribution.

        :return: None
        :rtype: None
        """
        self.static_value = self.distribution.get_random_value()

    def get_prob(self) -> Union[None, Dict[str, float]]:
        """
        Return probabilities of values occurring. Probabilities are calculated based on occurrence counters

        :return: Dictionary, where each key is possible value in distribution and key is probability of it occuring
        :rtype: Union(None, Dict[str, float])
        """
        if self.counter:
            total_occurrences = sum(self.counter.values())
            return {key: value / float(total_occurrences) for key, value in self.counter.items()}
        else:
            return None

    def is_value_possible(self, value: str):
        """
        Checks if given value is possible in Nodes distribution

        :param value: Value to check
        :type value: str
        :return: True if value is possible in distribution and False otherwise
        :rtype: bool
        """
        return self.distribution.is_value_possible(value)


class BayesNetwork:
    """
    Class representing Bayes Netowrk
    """

    def __init__(self):
        self.nodes = {}

    def add_nodes(self, list_of_nodes: List['Node']) -> None:
        """
        Adds nodes to network.

        :param list_of_nodes: List of nodes
        :type list_of_nodes: List['Node']
        :return: None
        :rtype: None
        """
        assert isinstance(list_of_nodes, List) and all(isinstance(x, Node) for x in list_of_nodes)
        for state in list_of_nodes:
            if state.name not in self.nodes.keys():
                self.nodes[state.name] = state
            else:
                raise ValueError('Nodes names must be unique')

    def add_edge(self, parent: 'Node', child: 'Node') -> None:
        """
        Adds edge between given two nodes if they are already in network.

        :param parent: Parent node
        :type parent: 'Node'
        :param child: Child node
        :type child: 'Node'
        :return: None
        :rtype: None
        """
        assert isinstance(parent, Node) and isinstance(child, Node)

        if parent.name not in self.nodes.keys():
            raise ValueError('Not known parent node {}, it first must be added to list of stated'.format(parent))

        if child.name not in self.nodes.keys():
            raise ValueError('Not known child node {}, it first must be added to list of stated'.format(child))

        parent.add_child(child)
        child.add_parent(parent)

    def preprocess(self):
        """
        This method must be called before performing any calculations. It performs necessary internal preprocessing.

        :return: None
        :rtype: None
        """
        assert self.nodes, 'No nodes in graph'
        for node in self.nodes.values():
            node.preprocess()

    def _set_evidences(self, evidence: Dict[str, str]):
        """
        Set evidences.

        :param evidence: Evidence in form of dictionary, where keys are nodes names and values are values to be set
        in those nodes.
        :type evidence: Dict[str, str]
        :return:
        :rtype:
        """
        for name, state in evidence.items():
            self.nodes[name].set_evidence(state)

    def _check_query(self, query):
        """
        Checks if Nodes in query are in network

        :param query: List of nodes names
        :type query: List[str]
        :return: None
        :rtype: None
        """
        # TODO change nodes to nodes names
        for node_name in query:
            assert node_name in self.nodes.keys(), 'Node {} not known'.format(node_name)

    def _reset_counters(self):
        """
        Resets occurrence counters in each node in network

        :return:
        :rtype:
        """
        assert self.nodes
        for state in self.nodes.values():
            state.reset_counters()

    def _get_nodes_without_evidence(self, evidence: Dict[str, str]):
        """
        Returns all nodes without given evidence

        :param evidence:  Evidence in form of dictionary, where keys are nodes names and values are values to be set
        in those nodes.
        :type evidence:  Dict[str, str]
        :return: List of nodes without evidences set
        :rtype: List['Node']
        """
        nodes_without_evidence = []
        for state in self.nodes.values():
            if state.name not in evidence.keys():
                nodes_without_evidence.append(state)

        return nodes_without_evidence

    def gibbs(self, evidence: Dict[str, str], query: List[str], n: int) -> Dict[str, Dict[str, float]]:
        """
        Performc MCMC Gibbs sampling

        :param evidence:  Evidence in form of dictionary, where keys are nodes names and values are values to be set
        in those nodes.
        :type evidence:Dict[str, str]
        :param query: Names of nodes for which probabilities should be approximated
        :type query: List[str]
        :param n: Number of loop iterations used in Gibbs sampling
        :type n: int
        :return: Dictionary where keys are nodes names and values are dictionaries with probabilities descriptions.
        :rtype: Dict[str,Dict[str, float]]
        """
        assert self.nodes, 'No nodes added to network'

        for node_name in query:
            if node_name in evidence.keys():
                raise ValueError('Node {} is in query as well as in evidence'.format(node_name))

        self._check_query(query)
        self._set_evidences(evidence)
        self._reset_counters()

        nodes_without_evidence = self._get_nodes_without_evidence(evidence)
        if not nodes_without_evidence:
            raise ValueError('Every node was given evidence, cant generate any data')

        # setting random initial values
        for node in nodes_without_evidence:
            node.set_random_initial_value()

        # Monte Carlo simulation
        for i in range(n):
            node = random_choice(nodes_without_evidence)
            markov_blanket = node.get_markov_blanket()

            node.set_non_static()
            value = node.sample_given_markov_blanket(markov_blanket)
            node.set_static_value(value)

        results = {}
        for node_name in query:
            results[node_name] = self.nodes[node_name].get_prob()

        return results


1
