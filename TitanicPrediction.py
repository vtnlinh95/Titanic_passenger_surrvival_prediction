import operator
import pandas as pd
import math as math

# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()

def accuracy_score(truth, pred, name):
    """ Returns accuracy score for input truth and predictions. """

    if len(truth) == len(pred):
        # Calculate and return the accuracy as a percent
        # {:2f}.format((truth == pred).mean()*100)
        # ":" represents format specification
        # "2f" represents 2 decimal places
        return "{} Predictions have an accuracy of {:.2f}.".format(name, (truth.values == pred.values).mean() * 100)
    else:
        return "Number of predictions does not match number of outcomes!"


def calculate_entropy(original_data, feature, value):
    data = original_data.copy()
    if feature != None and value != None:
        data = data[data[feature] == value]
    # count row
    # data.shape[1] count columns
    total = (float)(data.shape[0])
    num_survived = (float)(data[data['Survived'] == 1].shape[0])
    num_not_survived = (float)(total - num_survived)
    if num_survived == 0.0 or num_not_survived == 0.0:
        return 0
    pyes = -(num_survived / total) * math.log(num_survived / total, 2)
    pno = -(num_not_survived / total) * math.log(num_not_survived / total, 2)
    pt = total / original_data.shape[0]
    return (pyes + pno) * pt


def find_tree(original_data):
    count_survived = original_data[original_data['Survived'] == 1].shape[0]
    count_not_survived = original_data[original_data['Survived'] == 0].shape[0]
    if count_survived == 0 and count_not_survived != 0:
        print "pure_leaf Not Survied"
        return 0
    if count_not_survived == 0 and count_survived != 0:
        print "pure_leaf Survied"
        return 1
    list_features = list(original_data.columns.values)
    list_features.remove('Survived')
    max_feature = None
    max_gain = -1
    for feature in list_features:
        list_values = original_data[feature].unique()
        entropy = 0
        for value in list_values:
            entropy += calculate_entropy(original_data, feature, value)
        gain = calculate_entropy(original_data, feature=None, value=None) - entropy
        if gain > max_gain:
            max_feature = feature
            max_gain = gain
    tree = {max_feature: {}}
    for value in original_data[max_feature].unique():
        new_data = original_data.copy()
        new_data = new_data[new_data[max_feature] == value]
        new_data = new_data.drop(max_feature, axis=1)
        features = list(new_data.columns.values)
        if len(features) == 1 and features.__contains__('Survived'):
            if count_not_survived > count_survived:
                print "Not Survived"
                tree[max_feature][value] = 0
            else:
                print "Survived"
                tree[max_feature][value] = 1
            continue
        print max_feature
        print value
        subtree = find_tree(new_data)
        tree[max_feature][value] = subtree
    return tree

# Load the dataset
in_file = 'train.csv'
full_data = pd.read_csv(in_file)
full_data = full_data.drop('PassengerId', axis=1)
full_data = full_data.drop('Ticket', axis=1)
full_data = full_data.drop('Pclass', axis=1)
full_data = full_data.drop('Age', axis=1)
full_data = full_data.drop('Sex', axis=1)

full_data['Cabin'].fillna('U', inplace=True)

full_data.loc[operator.__and__(0.0 <= full_data.Fare, full_data.Fare <= 100.0), 'Fare'] = 1
full_data.loc[operator.__and__(101.0 <= full_data.Fare, full_data.Fare <= 200.0), 'Fare'] = 2
full_data.loc[operator.__and__(201.0 <= full_data.Fare, full_data.Fare <= 300.0), 'Fare'] = 3
full_data.loc[operator.__and__(301.0 <= full_data.Fare, full_data.Fare <= 400.0), 'Fare'] = 4
full_data.loc[operator.__and__(401.0<= full_data.Fare, full_data.Fare <= 500.0), 'Fare'] = 5
full_data.loc[501.0 <= full_data.Fare, 'Fare'] = 6

full_data['Name'] = full_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
full_data['Name'] = full_data.Name.map(Title_Dictionary)

full_data['Cabin'] = full_data['Cabin'].apply(lambda cabin : cabin[:1])

original_data = full_data.sample(frac=0.7)
test_data = full_data.drop(original_data.index)
# test_data = test_data[test_data['Age'] == 3]
tree = {}
outcomes = test_data['Survived']
original_data.head()
tree = find_tree(original_data)

results = []
for entry in test_data.iterrows():
    tempDict = tree.copy()
    result = ""
    while (isinstance(tempDict, dict)):
        root = Node(tempDict.keys()[0], tempDict[tempDict.keys()[0]])
        tempDict = tempDict[tempDict.keys()[0]]
        value = entry[1][root.value]
        if (value in tempDict.keys()):
            child = Node(value, tempDict[value])
            result = tempDict[value]
            tempDict = tempDict[value]
        else:
            result = "Null"
            break
    if result != "Null":
        results.append(result)
    else:
        results.append(0)
results = pd.Series(results)
outcomes = test_data['Survived']
results.reset_index(drop=True)
outcomes.reset_index(drop=True)
print(accuracy_score(outcomes, results, "Accuracy"))
