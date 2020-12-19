########################################################################################################################
__author__ = 'Piotr Kramek'

########################################################################################################################
if __name__ == '__main__':
    from pprint import pprint

    from numpy import random

    from BayesNetwork.bayesNetwork import Node, BayesNetwork
    from BayesNetwork.distributions import DiscreteDistribution, ConditionalDistribution

    # Random seed initialization
    random.seed(42)

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
    hospitalized_node = Node(hospitalization_dist, name='Hospitalized')

    network = BayesNetwork()

    network.add_nodes(
        [fever_node, fatigue_node, shortness_of_breath_node, coronavirus_node, test_node, hospitalized_node]
    )

    # Order is important
    network.add_edge(fever_node, coronavirus_node)
    network.add_edge(fatigue_node, coronavirus_node)
    network.add_edge(shortness_of_breath_node, coronavirus_node)

    network.add_edge(coronavirus_node, test_node)

    network.add_edge(coronavirus_node, hospitalized_node)
    network.add_edge(test_node, hospitalized_node)

    network.preprocess()
    ####################################################################################################################

    # What is the probability distribution of need of hospitalization when patient has fever,
    # feels fatigued and has a shortness of breath
    evidences = {'Fever': 'fever', 'Fatigue': 'fatigue',
                 'Shortness of breath': 'shortness of breath'}
    query = [hospitalized_node.name]
    results = network.gibbs(evidences, query, 10000)
    pprint(results)

    # What are the probability distributions that test will be positive and patient is really sick given that patient
    # has fever, and shortness of breath but does not feel fatigue
    evidences = {'Fever': 'fever', 'Fatigue': 'no fatigue',
                 'Shortness of breath': 'shortness of breath'}
    query = [coronavirus_node.name, test_node.name]
    results = network.gibbs(evidences, query, 10000)
    pprint(results)

    # What is probability distribution that patient is sick given that he only has fever
    evidences = {'Fever': 'fever', 'Fatigue': 'no fatigue',
                 'Shortness of breath': 'no shortness of breath'}
    query = [coronavirus_node.name]
    results = network.gibbs(evidences, query, 10000)
    pprint(results)

    # What is a probability distribution that patient will be hospitalized if he is not really sick
    evidences = {'Coronavirus': 'not sick'}
    query = [hospitalized_node.name]
    results = network.gibbs(evidences, query, 10000)
    pprint(results)
