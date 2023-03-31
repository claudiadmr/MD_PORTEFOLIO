from typing import List, Set, Dict

class TransactionDataset:
    def __init__(self, transactions: List[Set[int]]):
        # transactions is a list of sets, where each set represents a transaction
        self.transactions = transactions
        # find frequent items in the transaction dataset
        self.freq_items = self._find_frequent_items()

    def _find_frequent_items(self):
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction:
                # increment count for item if it already exists in item_counts, otherwise initialize count to 1
                if item in item_counts:
                    item_counts[item] += 1
                else:
                    item_counts[item] = 1
        # sort item_counts by count in descending order and return as a list of (item, count) tuples
        freq_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return freq_items



class Apriori:
    def __init__(self, dataset: TransactionDataset, min_support: float):
        self.dataset = dataset
        self.min_support = min_support
        # generate frequent itemsets using the Apriori algorithm
        self.freq_itemsets = self._generate_frequent_itemsets()

    def _generate_frequent_itemsets(self):
        freq_itemsets = {}
        # generate 1-itemsets and add them to freq_itemsets
        one_itemsets = {frozenset([item]): count for item, count in self.dataset.freq_items}
        freq_itemsets.update(one_itemsets)

        k = 2
        while True:
            # generate candidate k-itemsets from frequent (k-1)-itemsets
            k_itemsets = self._generate_candidate_itemsets(freq_itemsets, k)
            if not k_itemsets:
                break
            # find frequent k-itemsets
            k_freq_itemsets = self._get_frequent_itemsets(k_itemsets)
            if not k_freq_itemsets:
                break
            # add frequent k-itemsets to freq_itemsets
            freq_itemsets.update(k_freq_itemsets)
            k += 1

        return freq_itemsets

    def _generate_candidate_itemsets(self, freq_itemsets, k):
        candidates = set()
        # generate candidate k-itemsets by joining frequent (k-1)-itemsets
        for itemset1 in freq_itemsets:
            for itemset2 in freq_itemsets:
                if len(itemset1.union(itemset2)) == k:
                    candidates.add(itemset1.union(itemset2))
        return candidates

    def _get_frequent_itemsets(self, itemsets):
        item_counts = {}
        # count occurrences of itemsets in transactions
        for transaction in self.dataset.transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset in item_counts:
                        item_counts[itemset] += 1
                    else:
                        item_counts[itemset] = 1
        # find frequent itemsets whose support is greater than or equal to min_support
        freq_itemsets = {itemset: count for itemset, count in item_counts.items() if
                         count / len(self.dataset.transactions) >= self.min_support}
        return freq_itemsets


class AssociationRules:
    def __init__(self, apriori: Apriori, min_confidence: float):
        self.apriori = apriori
        self.min_confidence = min_confidence
        # generate association rules using the Apriori algorithm
        self.rules = self._generate_association_rules()

    def _generate_association_rules(self):
        rules = []
        # Iterate over frequent itemsets found by Apriori algorithm
        for itemset in self.apriori.freq_itemsets.keys():
            # Check if the itemset has more than one item
            if len(itemset) > 1:
                # Iterate over each item in the itemset
                for item in itemset:
                    # Define antecedent as a frozen set with the current item
                    antecedent = frozenset([item])
                    # Define consequent as the set difference between the itemset and the antecedent
                    consequent = itemset - antecedent
                    # Calculate confidence of the rule using the support of the itemset and antecedent
                    confidence = self.apriori.freq_itemsets[itemset] / self.apriori.freq_itemsets[antecedent]
                    # Check if the confidence is greater than or equal to the minimum confidence threshold
                    if confidence >= self.min_confidence:
                        # Append the rule to the list of rules
                        rules.append((antecedent, consequent, confidence))
        # Return the list of generated association rules
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


'''
The code contains implementations of the Apriori algorithm for frequent itemset mining and association rule mining. 
It consists of three classes:

1. TransactionDataset: This class is used to store a dataset of transactions as a list of sets, where each set 
contains the items present in the transaction. It has a private method _find_frequent_items() that finds the frequent 
items in the dataset, and stores them in a dictionary where the keys are the frequent items and the values are their 
support counts.

2. Apriori: This class is used to generate frequent itemsets from the given transaction dataset using the Apriori 
algorithm. It takes a TransactionDataset object and a min_support threshold as input. It has a private method 
_generate_frequent_itemsets() that generates frequent itemsets of varying sizes, and stores them in a dictionary 
where the keys are the itemsets and the values are their support counts. It also has two additional private methods: 
_generate_candidate_itemsets() generates candidate itemsets of a given size, and _get_frequent_itemsets() filters the 
candidate itemsets to get the frequent itemsets.

3. AssociationRules: This class is used to generate association rules from the frequent itemsets generated by the 
Apriori class. It takes an Apriori object and a min_confidence threshold as input. It has a private method 
_generate_association_rules() that generates association rules from the frequent itemsets, and stores them as a list 
of tuples where each tuple contains the antecedent, consequent, and confidence of the rule.

The code also includes two test functions test1() and test2() that demonstrate how to use the TransactionDataset, 
Apriori, and AssociationRules classes to generate frequent itemsets and association rules from sample datasets.

To use the code, you can create a TransactionDataset object with a list of transaction sets, set the min_support 
threshold and generate frequent itemsets using Apriori, set the min_confidence threshold and generate association 
rules using AssociationRules. Finally, you can access the frequent itemsets and association rules generated by these 
classes and use them for further analysis or processing.
'''
