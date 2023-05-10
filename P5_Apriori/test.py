import unittest
from os import name

# Import the classes to be tested
from apriori import TransactionDataset, Apriori, AssociationRules

class TestTransactionDataset(unittest.TestCase):
    def test_frequent_items(self):
        # Create a TransactionDataset object
        dataset = TransactionDataset([
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ])

        # Test the frequent_items() method
        expected = [            {3, 2, 1},            {3, 2},            {3, 1},            {3},            {2},            {1}        ]
        self.assertEqual(dataset.frequent_items(), expected)

class TestApriori(unittest.TestCase):
    def test_freq_itemsets(self):
        # Create a TransactionDataset object
        dataset = TransactionDataset([
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ])

        # Create an Apriori object
        apriori = Apriori(dataset, min_support=0.5)

        # Test the freq_itemsets attribute
        expected = {
            frozenset({3}): 7,
            frozenset({2}): 6,
            frozenset({1}): 5,
            frozenset({2, 3}): 5,
            frozenset({1, 2}): 4,
            frozenset({1, 3}): 4,
            frozenset({3, 5}): 3,
            frozenset({1, 2, 3}): 3,
            frozenset({4}): 3,
            frozenset({1, 5}): 2,
            frozenset({2, 4}): 2,
            frozenset({3, 4}): 2,
            frozenset({1, 2, 5}): 2,
            frozenset({2, 3, 5}): 2
        }
        self.assertEqual(apriori.freq_itemsets, expected)

class TestAssociationRules(unittest.TestCase):
    def test_rules(self):
        # Create a TransactionDataset object
        dataset = TransactionDataset([
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ])

        # Create an Apriori object
        apriori = Apriori(dataset, min_support=0.5)

        # Create an AssociationRules object
        rules = AssociationRules(apriori, min_confidence=0.8)

        # Test the rules() method
        expected = [        ({3}, {1}, 0.5714285714285714),        ({3}, {2}, 0.7142857142857143),        ({2, 3}, {1}, 0.6),        ({1, 3}, {2}, 1.0),        ({3, 5}, {2}, 0.6666666666666666),        ({2, 3, 5}, {1}, 1.0)    ]
        for i, rule in enumerate(rules.rules()):
            self.assertEqual(rule, expected[i])


if name == 'main':
    unittest.main()