from typing import List, Set, Dict


class TransactionDataset:
    def __init__(self, transactions: List[Set[int]]):
        self.transactions = transactions
        self.freq_items = self._find_frequent_items()

    def _find_frequent_items(self):
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction:
                if item in item_counts:
                    item_counts[item] += 1
                else:
                    item_counts[item] = 1
        freq_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return freq_items


class Apriori:
    def __init__(self, dataset: TransactionDataset, min_support: float):
        self.dataset = dataset
        self.min_support = min_support
        self.freq_itemsets = self._generate_frequent_itemsets()

    def _generate_frequent_itemsets(self):
        freq_itemsets = {}
        one_itemsets = {frozenset([item]): count for item, count in self.dataset.freq_items}
        freq_itemsets.update(one_itemsets)

        k = 2
        while True:
            k_itemsets = self._generate_candidate_itemsets(freq_itemsets, k)
            if not k_itemsets:
                break
            k_freq_itemsets = self._get_frequent_itemsets(k_itemsets)
            if not k_freq_itemsets:
                break
            freq_itemsets.update(k_freq_itemsets)
            k += 1

        return freq_itemsets

    def _generate_candidate_itemsets(self, freq_itemsets, k):
        candidates = set()
        for itemset1 in freq_itemsets:
            for itemset2 in freq_itemsets:
                if len(itemset1.union(itemset2)) == k:
                    candidates.add(itemset1.union(itemset2))
        return candidates

    def _get_frequent_itemsets(self, itemsets):
        item_counts = {}
        for transaction in self.dataset.transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset in item_counts:
                        item_counts[itemset] += 1
                    else:
                        item_counts[itemset] = 1

        freq_itemsets = {itemset: count for itemset, count in item_counts.items() if
                         count / len(self.dataset.transactions) >= self.min_support}
        return freq_itemsets


class AssociationRules:
    def __init__(self, apriori: Apriori, min_confidence: float):
        self.apriori = apriori
        self.min_confidence = min_confidence
        self.rules = self._generate_association_rules()

    def _generate_association_rules(self):
        rules = []
        for itemset in self.apriori.freq_itemsets.keys():
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = frozenset([item])
                    consequent = itemset - antecedent
                    confidence = self.apriori.freq_itemsets[itemset] / self.apriori.freq_itemsets[antecedent]
                    if confidence >= self.min_confidence:
                        rules.append((antecedent, consequent, confidence))
        return rules

def test1():
    # create a sample transaction dataset
    transactions = [
        {1, 2, 3},
        {2, 3, 4},
        {1, 2, 3, 4},
        {2, 3},
        {1, 3, 4},
        {1, 2},
        {1, 3},
        {1, 2, 3, 4},
        {1, 2, 3},
        {1, 2, 4},
        {1, 2, 3, 4},
        {2, 4},
        {2, 3, 4},
        {1, 2, 4},
        {1, 2, 3},
    ]

    # create a TransactionDataset object
    dataset = TransactionDataset(transactions)

    # set minimum support and generate frequent itemsets using Apriori
    min_support = 0.3
    apriori = Apriori(dataset, min_support)

    # print frequent itemsets
    print("Frequent Itemsets:")
    for itemset, support in apriori.freq_itemsets.items():
        print(f"{itemset}: {support}")

    # set minimum confidence and generate association rules
    min_confidence = 0.5
    rules = AssociationRules(apriori, min_confidence)

    # print association rules
    print("Association Rules:")
    for antecedent, consequent, confidence in rules.rules:
        print(f"{antecedent} -> {consequent}: {round(confidence,2)}")

def test2():
    # create a sample transaction dataset
    transactions = [
        {1, 2, 3},
        {1, 2, 4},
        {1, 2},
        {1, 3},
        {2, 3},
        {2, 4},
        {3, 4},
        {1, 3, 4},
        {2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3},
        {1, 2, 4},
        {1, 3, 4},
        {2, 3, 4},
        {1, 2, 3, 4},
    ]

    # create a TransactionDataset object
    dataset = TransactionDataset(transactions)

    # set minimum support and generate frequent itemsets using Apriori
    min_support = 0.4
    apriori = Apriori(dataset, min_support)

    # print frequent itemsets
    print("Frequent Itemsets:")
    for itemset, support in apriori.freq_itemsets.items():
        print(f"{itemset}: {support}")

    # set minimum confidence and generate association rules
    min_confidence = 0.6
    rules = AssociationRules(apriori, min_confidence)

    # print association rules
    print("Association Rules:")
    for antecedent, consequent, confidence in rules.rules:
        print(f"{antecedent} -> {consequent}:  { round(confidence,2)}")


if __name__ == "__main__":
    test1()
    print("\n")
    test2()
#%%
