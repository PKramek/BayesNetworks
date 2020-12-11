from BayesNetwork.bayesNetwork import Node, BayesNetwork
from BayesNetwork.distributions import DiscreteDistribution, ConditionalDistribution

first_dist = DiscreteDistribution({'A': 0.5, 'B': 0.5})
second_dist = DiscreteDistribution({'C': 0.3, 'D': 0.7})

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

node_1 = Node(first_dist, name='First')
node_2 = Node(second_dist, name='Second')
node_3 = Node(conditional_dist, name='Conditional')

network = BayesNetwork()

network.add_states([node_1, node_2, node_3])
network.add_edge(node_1, node_3)
network.add_edge(node_2, node_3)
network.preprocess()

for i in range(1000):
    evidence = [node_1.sample(), node_2.sample()]
    node_3.sample(evidence)

print(node_1.get_prob())
print(node_2.get_prob())
print(node_3.get_prob())

network.mcmc({'First': 'A', 'Second': 'C', 'Conditional': 'F'}, 1000)


def gibbs(bayes_network: BayesNetwork, evidence: dict, query):
    pass
