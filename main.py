from pprint import pprint

from BayesNetwork.bayesNetwork import Node, BayesNetwork
from BayesNetwork.distributions import DiscreteDistribution, ConditionalDistribution

fever_dist = DiscreteDistribution({'fever': 0.05, 'no fever': 0.95})
fatigue_dist = DiscreteDistribution({'fatigue': 0.3, 'no fatigue': 0.7})
shortness_of_breath_dist = DiscreteDistribution({'shortness of breath': 0.01, 'no shortness of breath': 0.99})

sick_dist = ConditionalDistribution([
    ['fever', 'fatigue', 'shortness of breath', 'sick', 0.8],
    ['fever', 'fatigue', 'shortness of breath', 'not sick', 0.2],
    ['fever', 'fatigue', 'no shortness of breath', 'sick', 0.6],
    ['fever', 'fatigue', 'no shortness of breath', 'not sick', 0.4],
    ['fever', 'no fatigue', 'shortness of breath', 'sick', 0.5],
    ['fever', 'no fatigue', 'shortness of breath', 'not sick', 0.5],
    ['fever', 'no fatigue', 'no shortness of breath', 'sick', 0.4],
    ['fever', 'no fatigue', 'no shortness of breath', 'not sick', 0.6],
    ['no fever', 'fatigue', 'shortness of breath', 'sick', 0.75],
    ['no fever', 'fatigue', 'shortness of breath', 'not sick', 0.25],
    ['no fever', 'fatigue', 'no shortness of breath', 'sick', 0.2],
    ['no fever', 'fatigue', 'no shortness of breath', 'not sick', 0.8],
    ['no fever', 'no fatigue', 'shortness of breath', 'sick', 0.3],
    ['no fever', 'no fatigue', 'shortness of breath', 'not sick', 0.7],
    ['no fever', 'no fatigue', 'no shortness of breath', 'sick', 0.01],
    ['no fever', 'no fatigue', 'no shortness of breath', 'not sick', 0.99],
])

test_dist = ConditionalDistribution([
    ['sick', 'positive', 0.9],
    ['sick', 'negative', 0.1],
    ['not sick', 'positive', 0.05],
    ['not sick', 'negative', 0.95]
])

hospitalization_dist = ConditionalDistribution([
    ['sick', 'positive', 'hospitalized', 0.6],
    ['sick', 'positive', 'not hospitalized', 0.4],
    ['sick', 'negative', 'hospitalized', 0],
    ['sick', 'negative', 'not hospitalized', 1],
    ['not sick', 'positive', 'hospitalized', 0.1],
    ['not sick', 'positive', 'not hospitalized', 0.9],
    ['not sick', 'negative', 'hospitalized', 0.0],
    ['not sick', 'negative', 'not hospitalized', 1.0],

])
fever_node = Node(fever_dist, name='Fever')
fatigue_node = Node(fatigue_dist, name='Fatigue')
shortness_of_breath_node = Node(shortness_of_breath_dist, name='Shortness of breath')
test_node = Node(test_dist, name='Test')
coronavirus_node = Node(sick_dist, name='Coronavirus')
hospitalization_node = Node(hospitalization_dist, name='Hospitalized')

network = BayesNetwork()

network.add_nodes(
    [fever_node, fatigue_node, shortness_of_breath_node, coronavirus_node, test_node, hospitalization_node]
)

# Order is important
network.add_edge(fever_node, coronavirus_node)
network.add_edge(fatigue_node, coronavirus_node)
network.add_edge(shortness_of_breath_node, coronavirus_node)

network.add_edge(coronavirus_node, test_node)

network.add_edge(coronavirus_node, hospitalization_node)
network.add_edge(test_node, hospitalization_node)

network.preprocess()

evidences = {'Fever': 'fever', 'Fatigue': 'fatigue',
             'Shortness of breath': 'shortness of breath'}
query = [test_node.name]
results = network.gibbs(evidences, query, 10000)

pprint(results)
