from pprint import pprint

from BayesNetwork.bayesNetwork import Node, BayesNetwork
from BayesNetwork.distributions import DiscreteDistribution, ConditionalDistribution

first_dist = DiscreteDistribution({'A': 0.1, 'B': 0.9})
second_dist = DiscreteDistribution({'C': 0.2, 'D': 0.8})
third_dist = DiscreteDistribution({'X': 0.2, 'Y': 0.8})
fourth_dist = DiscreteDistribution({'True': 0.1, 'False': 0.9})

first_dist.preprocess()
first_dist.sample(10)

test_conditional = ConditionalDistribution([
    ['A', 'X', 0.5],
    ['A', 'Y', 0.5],
    ['B', 'X', 0.3],
    ['B', 'Y', 0.7]
])
test_conditional.preprocess()

conditional_dist = ConditionalDistribution([
    ['A', 'C', 'E', 0.5],
    ['A', 'C', 'F', 0.5],
    ['A', 'D', 'E', 0.3],
    ['A', 'D', 'F', 0.7],
    ['B', 'C', 'E', 0.3],
    ['B', 'C', 'F', 0.7],
    ['B', 'D', 'E', 0.5],
    ['B', 'D', 'F', 0.5],
]
)

second_conditional = ConditionalDistribution([
    ['C', 'X', 'G', 0.5],
    ['C', 'X', 'H', 0.5],
    ['C', 'Y', 'G', 0.3],
    ['C', 'Y', 'H', 0.7],
    ['D', 'X', 'G', 0.3],
    ['D', 'X', 'H', 0.7],
    ['D', 'Y', 'G', 0.5],
    ['D', 'Y', 'H', 0.5],
]
)

node_1 = Node(first_dist, name='First')
node_2 = Node(second_dist, name='Second')
node_3 = Node(conditional_dist, name='Conditional')
node_4 = Node(third_dist, name='Third')
node_5 = Node(second_conditional, name='Second Conditional')

not_know_node = Node(fourth_dist, name='temp')

network = BayesNetwork()

network.add_states([node_1, node_2, node_3, node_4, node_5])

network.add_edge(node_1, node_3)
network.add_edge(node_2, node_3)
network.add_edge(node_2, node_5)
network.add_edge(node_4, node_5)
network.preprocess()

results = network.gibbs({node_3.name: 'E', node_4.name: 'X'}, [node_1, node_2, node_5], 10000)
pprint(results)
