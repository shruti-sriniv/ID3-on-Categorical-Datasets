import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to calculate entropy
# entropy(s) = -P(1) * log2(P(1)) - p(0) *log2(P(0))


def total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]  # the total size of the dataset
    total_entr = 0

    for c in class_list:  # for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0]  # number of the class
        total_class_entr = - (total_class_count / total_row) * np.log2(
            total_class_count / total_row)  # entropy of the class
        total_entr += total_class_entr  # adding the class entropy to the total entropy of the dataset

    return total_entr

# Now we calculate the entropy for each individual column

def indiv_entropy(feature, label, class_labels):
    class_count = feature.shape[0]
    entropy = 0

    for bin in class_labels:
        label_class_count = feature[feature[label] == bin].shape[0]  # row count of the binary
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count  # probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  # entropy
        entropy += entropy_class

    return entropy

# Calculating the information gain for each feature
# The formula for information gain if there are three values of PClass is:
# I(PClass) = p(PClass = 1) * Entropy(Survived =1) + p(PClass = 1) * Entropy(Survived = 1) ....


def infor_gain(feature, train_data, label, class_labels):
    uniq_feat_vals = train_data[feature].unique()
    rows = train_data.shape[0]
    info = 0.0

    for feat in uniq_feat_vals:
        feat_val_data = train_data[train_data[feature] == feat]
        feat_val_count = feat_val_data.shape[0]
        feat_val_entropy = indiv_entropy(feat_val_data, label, class_labels)
        feat_val_prob = feat_val_count/rows
        info += feat_val_prob * feat_val_entropy

    return total_entropy(train_data, label, class_labels) - info

# Finding the feature with the most information gain
# This will be the first feature we split based on


def find_best_feat(train_data, label, class_labels):
    max_info_gain = -1
    max_info_feature = None

    for feature in train_data.columns.drop(label):
        info_gain = infor_gain(feature, train_data, label, class_labels)
        if max_info_gain < info_gain:
            max_info_gain = info_gain
            max_info_feature = feature

    print(max_info_feature)
    return max_info_feature

# Creating a sub-tree of a feature and removing the value of that feature from the dataset


def sub_tree(feature, train_data, label, class_labels):
    tree = {}
    feature_val_count_dict = train_data[feature].value_counts(sort=False)

    for feat_val, count in feature_val_count_dict.items():
        assigned = False
        feat_val_data = train_data[train_data[feature] == feat_val]
        for bin in class_labels:
            class_count = feat_val_data[feat_val_data[label] == bin].shape[0]

            if class_count == count:
                tree[feat_val] = bin
                train_data = train_data[train_data[feature] != feat_val]
                assigned = True
        if not assigned:
            tree[feat_val] = '?'
    return tree, train_data

# Creating the tree using the features that we have split based


def create_tree(root_node, prev_feature_val, train_data, label, class_labels):
    if train_data.shape[0] != 0: # Checking to see if the dataset is empty
        max_info_feature = find_best_feat(train_data, label, class_labels)
        tree, train_data = sub_tree(max_info_feature, train_data, label, class_labels)
        next_root = None

        if prev_feature_val != None:
            root_node[prev_feature_val] = dict()
            root_node[prev_feature_val][max_info_feature] = tree
            next_root = root_node[prev_feature_val][max_info_feature]

        else:
            root_node[max_info_feature] = tree
            next_root = root_node[max_info_feature]

        for node, branch in list(next_root.items()):
            if branch == '?':
                feat_val_data = train_data[train_data[max_info_feature] == node]
                create_tree(next_root, node, feat_val_data, label, class_labels)

# ID 3 algorithm implementation


def id3(train_data, label):
    tree = {}
    class_labels = train_data[label].unique()
    create_tree(tree, None, train_data, label, class_labels)
    return tree


# Make predictions based on this tree

def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feat_val = instance[root_node]
        if feat_val in tree[root_node]:
            return predict(tree[root_node][feat_val], instance)
        else:
            return None


def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows():#for each row in the dataset
        result = predict(tree, test_data_m.loc[index]) #predict the row
        if result == test_data_m[label][index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy

train_data = pd.read_csv('bank-additional-full.csv', sep=';')

# test_data = pd.read_csv('test.csv', delimiter=',')

train, test = train_test_split(train_data, test_size=0.3)
tree = id3(train, 'y')

accuracy = evaluate(tree, test, 'y')
print(accuracy)