import numpy as np
import operator
import pandas as pd
import math as math
import seaborn as sns
import random
from titanic_visualizations import survival_stats


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


def find_tree(root_entropy, original_data):
    count_survived = original_data[original_data['Survived'] == 1].shape[0]
    count_not_survived = original_data[original_data['Survived'] == 0].shape[0]
    if count_survived == 0:
        print "pure_leaf Survied"
        return
    if count_not_survived == 0:
        print "pure_leaf Not Survied"
        return
    list_features = list(original_data.columns.values)
    list_features.remove('Survived')
    max_feature = None
    max_gain = -1
    max_entropy = 0
    for feature in list_features:
        list_values = original_data[feature].unique()
        entropy = 0
        for value in list_values:
            entropy += calculate_entropy(original_data, feature, value)
        gain = root_entropy - entropy
        if gain > max_gain:
            max_feature = feature
            max_entropy = entropy
    for value in original_data[max_feature].unique():
        new_data = original_data.copy()
        new_data = new_data[new_data[max_feature] == value]
        new_data = new_data.drop(max_feature, axis=1)
        features = list(new_data.columns.values)
        if len(features) == 1 and features[0] == 'Survived':
            print "Can not decide"
            continue
        print max_feature
        print value
        find_tree(max_entropy, new_data)
    return


def predict_for_passenger_in_thrities(data):
    predictions = []
    for index, passenger in data.iterrows():
        if passenger['Pclass'] == 1:
            if passenger['Sex'] == 'male':
                if passenger['SibSp'] == 0:
                    if passenger['Fare'] == 1:
                        if passenger['Cabin'][:1] == 'B':
                            predictions.append(1)
                        elif passenger['Cabin'][:1] == 'E':
                            predictions.append(0)
                        elif passenger['Cabin'][:1] == 'A':
                            predictions.append(1)
                        else:
                            predictions.append(random.randint(0, 1))
                    elif passenger['Fare'] == '2':
                        predictions.append(1)
                    elif passenger['Fare'] == '6':
                        predictions.append(0)
                    else:
                        predictions.append(random.randint(0, 1))
                elif passenger['SibSp'] == 1:
                    predictions.append(0)
                else:
                    predictions.append(random.randint(0, 1))
            else:
                predictions.append(0)
        elif passenger['Pclass'] == 3:
            predictions.append(0)
        elif passenger['Pclass'] == 2:
            if passenger['Sex'] == 'male':
                if passenger['Embarked'] == 'C':
                    predictions.append(1)
                elif passenger['Embarked'] == 'S':
                    predictions.append(0)
                else:
                    predictions.append(random.randint(0, 1))
            else:
                predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)

# Load the dataset
in_file = 'train.csv'
full_data = pd.read_csv(in_file)
full_data = full_data.drop('PassengerId', axis=1)
full_data = full_data.drop('Name', axis=1)
full_data = full_data.drop('Ticket', axis=1)
full_data = full_data.dropna()
full_data.loc[operator.__and__(0.0 <= full_data.Age, full_data.Age <= 10.0), 'Age'] = 0
full_data.loc[operator.__and__(11.0 <= full_data.Age, full_data.Age <= 20.0), 'Age'] = 1
full_data.loc[operator.__and__(21.0 <= full_data.Age, full_data.Age <= 30.0), 'Age'] = 2
full_data.loc[operator.__and__(31.0 <= full_data.Age, full_data.Age <= 40.0), 'Age'] = 3
full_data.loc[operator.__and__(41.0 <= full_data.Age, full_data.Age <= 50.0), 'Age'] = 4
full_data.loc[operator.__and__(51.0 <= full_data.Age, full_data.Age <= 60.0), 'Age'] = 5
full_data.loc[operator.__and__(61.0 <= full_data.Age, full_data.Age <= 70.0), 'Age'] = 6
full_data.loc[operator.__and__(71.0 <= full_data.Age, full_data.Age <= 80.0), 'Age'] = 7

full_data.loc[operator.__and__(0.0 <= full_data.Fare, full_data.Fare <= 100.0), 'Fare'] = 1
full_data.loc[operator.__and__(101.0 <= full_data.Fare, full_data.Fare <= 200.0), 'Fare'] = 2
full_data.loc[operator.__and__(201.0 <= full_data.Fare, full_data.Fare <= 300.0), 'Fare'] = 3
full_data.loc[operator.__and__(301.0 <= full_data.Fare, full_data.Fare <= 400.0), 'Fare'] = 4
full_data.loc[operator.__and__(401.0<= full_data.Fare, full_data.Fare <= 500.0), 'Fare'] = 5
full_data.loc[501.0 <= full_data.Fare, 'Fare'] = 6

full_data['Cabin'] = full_data['Cabin'].apply(lambda cabin : cabin[:1])

original_data = full_data.sample(frac=0.7)
test_data = full_data.drop(original_data.index)
test_data = test_data[test_data['Age'] == 3]
outcomes = test_data['Survived']
original_data.head()
root_entropy = calculate_entropy(original_data, feature=None, value=None)
find_tree(root_entropy, original_data)
# predictions = predict_for_passenger_in_thrities(test_data)
# predictions.reset_index(drop=True)
# outcomes.reset_index(drop=True)
# print(accuracy_score(outcomes, predictions, "Passenger in Thirties"))
arr1 = np.arra
