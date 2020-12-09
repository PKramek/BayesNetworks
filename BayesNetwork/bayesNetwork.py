from typing import List


class Node:
    def __init__(self, distribution, name: str):
        self.childred = []
        self.parents = []
        self.markov_neighborhood = []
        self.distribution = distribution
        self.name = name

    def add_parent(self, parent):
        assert isinstance(parent, Node)
        self.parents.append(parent)
        self.markov_neighborhood.append(parent)

    def add_child(self, child):
        assert isinstance(child, Node)
        self.childred.append(child)
        self.markov_neighborhood.append(child)

    def get_markov_neighborhood(self):
        return self.markov_neighborhood

    def get_children(self):
        return self.childred

    def get_parents(self):
        return self.parents


class BayesNetwork:
    def __init__(self):
        self.states = {}
        self.edges = []

    def add_states(self, list_of_states: List[Node]):
        assert isinstance(list_of_states, List) and all(isinstance(x, Node) for x in list_of_states)
        for state in list_of_states:
            if state.name not in self.states:
                self.states[state.name] = state
            else:
                raise ValueError('States names must be unique')

    def add_edge(self, first: Node, second: Node):
        assert isinstance(first, Node) and isinstance(second, Node)

        if first not in self.states:
            raise ValueError('Not known first state {}, it first must be added to list of stated'.format(first))

        if second not in self.states:
            raise ValueError('Not known second state {}, it first must be added to list of stated'.format(second))

        # TODO check if the order is right
        first.add_child(second)
        second.add_parent(first)
