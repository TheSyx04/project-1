import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from util import load_data

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node.
            threshold: The threshold used for splitting at this node.
            left: The left child node.
            right: The right child node.
            gain: The gain of the split.
            value: If this node is a leaf node, this attribute represents the predicted value
                    for the target variable.
        """
        
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
        
class DecisionTree():
    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
        """
        
        self.min_samples = min_samples
        self.max_depth = max_depth
        
    def split_data(self, dataset, feature, threshold):
        """
        Splits the given dataset into two datasets based on the given feature and threshold.

        Parameters:
            dataset (ndarray): Input dataset.
            feature (int): Index of the feature to be split on.
            threshold (float): Threshold value to split the feature on.

        Returns:
            left_dataset (ndarray): Subset of the dataset with values less than or equal to the threshold.
            right_dataset (ndarray): Subset of the dataset with values greater than the threshold.
        """
        
        left_dataset = dataset[dataset[:, feature] <= threshold]
        right_dataset = dataset[dataset[:, feature] > threshold]
        
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        
        return left_dataset, right_dataset
    
    def entropy(self, y):
        """
        Computes the entropy of the given label values.

        Parameters:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """
        
        entropy = 0
        
        unique, counts = np.unique(y, return_counts=True)
        for count in counts:
            p = count / len(y)
            entropy -= p * np.log2(p)
        
        return entropy
    
    def information_gain(self, parent, left, right):
        """
        Computes the information gain from splitting the parent dataset into two datasets.

        Parameters:
            parent (ndarray): Input parent dataset.
            left (ndarray): Subset of the parent dataset after split on a feature.
            right (ndarray): Subset of the parent dataset after split on a feature.

        Returns:
            information_gain (float): Information gain of the split.
        """
        
        information_gain = 0
        parent_entropy, left_entropy, right_entropy = self.entropy(parent), self.entropy(left), self.entropy(right)
        weight_left, weight_right = len(left) / len(parent), len(right) / len(parent)
        information_gain = parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)
        
        return information_gain
    
    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the given dataset.

        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain, 
              left and right datasets.
        """
        
        best_split = {'gain': -1, 'feature': None, 'threshold': None}
        for feature in range(num_features):
            feature_values = dataset[:, feature]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left, right = self.split_data(dataset, feature, threshold)
                if len(left) and len(right) >= self.min_samples:
                    gain = self.information_gain(dataset, left, right)
                    if gain > best_split['gain']:
                        best_split['gain'] = gain
                        best_split['feature'] = feature
                        best_split['threshold'] = threshold
                        best_split['left'] = left
                        best_split['right'] = right
        
        return best_split
    
    def calculate_leaf_value(self, y):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        
        y = list(y)
        return max(set(y), key=y.count)
    
    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = np.shape(X)
        if n_samples > self.min_samples and current_depth <= self.max_depth:
            best_split = self.best_split(dataset, n_samples, n_features)
            if best_split['gain']:
                left = self.build_tree(best_split['left'], current_depth + 1)
                right = self.build_tree(best_split['right'], current_depth + 1)
                return Node(best_split['feature'], best_split['threshold'], left, right, best_split['gain'])
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector.

        Args:
        x (ndarray): The feature vector to predict the target value for.
        node (Node): The current node being evaluated.

        Returns:
        The predicted target value for the given feature vector.
        """
        
        if node.value is not None:
            return node.value
        else:
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
        
    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        
        predictions = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        np.array(predictions)
        return predictions
    

def main():
    X, y = load_data("data/processed/dataset_1.csv")
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtr = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
    dtr.fit(X_train, y_train)
    predictions = dtr.predict(X_test)
    print(f"Accuracy of sklearn model: {accuracy_score(y_test, predictions)}")
    
    dt = DecisionTree(min_samples=2, max_depth=2)
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    print(f"Accuracy of my model: {accuracy_score(y_test, predictions)}")
    
if __name__ == "__main__":
    main()