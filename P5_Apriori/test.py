import unittest
from apriori import TransactionDataset, Apriori, AssociationRules

class TestTransactionDataset(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ]
        self.dataset = TransactionDataset(self.transactions)

    def test_frequent_items(self):
        expected = [(3, 8), (1, 6), (2, 6), (4, 4), (5, 4)]
        self.assertEqual(self.dataset.freq_items, expected)


class TestApriori(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ]
        self.dataset = TransactionDataset(self.transactions)
        self.apriori = Apriori(self.dataset, min_support=0.5)

    def test_freq_itemsets(self):
        expected = {
            frozenset({3}): 8,
            frozenset({1}): 6,
            frozenset({2}): 6,
            frozenset({4}): 4,
            frozenset({5}): 4,
            frozenset({2, 3}): 6,
            frozenset({1, 2}): 4,
            frozenset({1, 3}): 6,
            frozenset({3, 4}): 4,
            frozenset({3, 5}): 4,
            frozenset({1, 2, 3}): 4
        }
        self.assertEqual(self.apriori.freq_itemsets, expected)


class TestAssociationRules(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            {1, 2, 3},
            {1, 2, 3, 4},
            {2, 3, 4},
            {2, 3, 5},
            {1, 2, 3, 5},
            {1, 3, 4},
            {1, 3, 5},
            {1, 2, 3, 4, 5}
        ]
        self.dataset = TransactionDataset(self.transactions)
        self.apriori = Apriori(self.dataset, min_support=0.5)
        self.rules = AssociationRules(self.apriori, min_confidence=0.8)

    def test_rules(self):
        expected = [
            (frozenset({2}), frozenset({3}), 1.0),
            (frozenset({1}), frozenset({3}), 1.0),
            (frozenset({4}), frozenset({3}), 1.0),
            (frozenset({5}), frozenset({3}), 1.0)
        ]
        self.assertEqual(self.rules.rules, expected)


if __name__ == '__main__':
    unittest.main()