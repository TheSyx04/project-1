import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from util import *
warnings.filterwarnings("ignore")

class MyLogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []
        
    def _sigmoid_function(self, z):
        """
        Compute the sigmoid of a number z

        Args:
            z (scalar): A number.

        Returns:
            g (scalar): sigmoid(z).
            
        """
        if z >= 0:
            g = 1 / (1 + np.exp(-z))
        else:
            g = np.exp(z) / (1 + np.exp(z))
            
        return g

    def _sigmoid(self, z):
        return np.array([self._sigmoid_function(value) for value in z])
    
    def compute_cost(self, y_true, y_pred, lambda_=None):
        """
        Computes the cost over all examples
        Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (ndarray Shape (m,))  target value 
        w : (ndarray Shape (n,))  values of parameters of the model      
        b : (scalar)              value of bias parameter of the model
        Returns:
        cost : (scalar) cost 
        
        """
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        cost = -np.mean(y_zero_loss + y_one_loss)
        
        return cost
    
    def compute_gradients(self, x, y_true, y_pred):

        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b
    
    def update_model_parameters(self, error_w, error_b, learning_rate):
        self.weights = self.weights - learning_rate * error_w
        self.bias = self.bias - learning_rate * error_b
        
    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            x_dot_weights = np.matmul(self.weights, X.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_cost(y, pred)
            error_w, error_b = self.compute_gradients(X, y, pred)
            self.update_model_parameters(error_w, error_b, self.lr)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)
            
    def predict(self, X):
        x_dot_weights = np.matmul(X, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]
    
def main():
    X, y = load_data("data/processed/dataset_1.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    my_lr = MyLogisticRegression(0.01, 1000)
    my_lr.fit(X_train, y_train)
    y_pred = my_lr.predict(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)
    
    print("My logistic regression accuracy: ", accuracy_score(y_test, y_pred))
    print("Sckit-learn logistic regression accuracy: ", accuracy_score(y_test, y_pred_sklearn))
    
if __name__ == "__main__":
    main()