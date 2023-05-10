import numpy as np

def apriori(transactions, min_support, k=2):
    # Find unique items in the transactions
    unique_items = np.unique(np.concatenate(transactions))

    # Initialize frequent itemsets
    freq_itemsets = {}

    # Generate frequent itemsets of size 1
    freq_itemsets[1] = {}
    for item in unique_items:
        count = np.sum([item in transaction for transaction in transactions])
        support = count / len(transactions)
        if support >= min_support:
            freq_itemsets[1][frozenset([item])] = support

    # Generate frequent itemsets of size > 1

    while True:
        # Generate candidate itemsets of size k
        candidate_itemsets = {}
        for itemset1, support1 in freq_itemsets[k-1].items():
            for itemset2, support2 in freq_itemsets[k-1].items():
                if itemset1 != itemset2:
                    candidate_itemset = itemset1.union(itemset2)
                    if len(candidate_itemset) == k and candidate_itemset not in candidate_itemsets:
                        candidate_itemsets[candidate_itemset] = 0

        # Count the support of each candidate itemset
        for transaction in transactions:
            for candidate_itemset in candidate_itemsets:
                if candidate_itemset.issubset(transaction):
                    candidate_itemsets[candidate_itemset] += 1

        # Prune candidate itemsets with support less than min_support
        freq_itemsets[k] = {}
        for candidate_itemset, support in candidate_itemsets.items():
            support = support / len(transactions)
            if support >= min_support:
                freq_itemsets[k][candidate_itemset] = support

        # Stop if no frequent itemsets of size k are found
        if len(freq_itemsets[k]) == 0:
            del freq_itemsets[k]
            break

        k += 1

    return freq_itemsets


def test1():
    print("Test 1")
    # Test the apriori function
    transactions = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'C'],
        ['A', 'B'],
        ['A', 'B', 'D'],
        ['B', 'C', 'D'],
        ['B', 'C'],
        ['B', 'D'],
        ['C', 'D'],
        ['A', 'C', 'D']
    ]
    min_support = 0.4
    freq_itemsets = apriori(transactions, min_support)

    # Print the frequent itemsets
    for k, itemsets in freq_itemsets.items():
        print('Frequent itemsets of size', k)
        for itemset, support in itemsets.items():
            print(itemset, 'with support', round(support, 2))


def test2():
    print("Test 2")
    # Test the apriori function with increasing k
    transactions = [
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'B', 'C', 'E', 'F'],
        ['A', 'B', 'D', 'F', 'G'],
        ['A', 'C', 'D', 'F', 'H'],
        ['B', 'C', 'D', 'E', 'F'],
        ['B', 'C', 'E', 'F', 'G'],
        ['B', 'D', 'F', 'G', 'H'],
        ['C', 'D', 'E', 'F', 'H'],
        ['A', 'B', 'C', 'D', 'F', 'G'],
        ['A', 'B', 'C', 'D', 'E', 'F', 'H'],
        ['B', 'C', 'D', 'E', 'F', 'G', 'H'],
        ['A', 'C', 'D', 'E', 'F', 'G', 'H']
    ]
    min_support = 0.4
    for k in range(1, 5):
        freq_itemsets = apriori(transactions, min_support, k)

        # Print the frequent itemsets
        print('Frequent itemsets of size', k)
        for itemset, support in freq_itemsets.items():
            print(itemset, 'with support', round(support, 2))



if __name__ == '__main__':
    test1()
    print("\n")
    test2()
