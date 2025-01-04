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
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=None):
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []
        
    def _sigmoid_function(self, z):
        """
        Compute the sigmoid of a number z

        Arg:
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
        """
        Apply the sigmoid function element-wise to the input array.

        Arg:
            z (ndarray): Input array of values to apply the sigmoid function to.

        Returns:
            numpy.ndarray: Array with the sigmoid function applied to each element.
        """
        
        return np.array([self._sigmoid_function(value) for value in z])
    
    def compute_cost(self, y_true, y_pred):
        """
        Computes the cost for logistic regression.
        
        Args:
            y_true (ndarray Shape (m,)): True binary labels.
            y_pred (ndarray Shape (m,)): Predicted probabilities.
        
        Returns:
            cost (float): The computed log loss.
        """
        
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        
        regularization = np.sum(self.weights**2) if self.lambda_ else 0
        
        cost = -np.mean(y_zero_loss + y_one_loss) + (self.lambda_/(2*len(y_true))) * regularization
        
        return cost
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        Compute the gradients of the loss function with respect to the weights and bias.
        
        Args:
            X (ndarray Shape (m,n)): The input features, with m examples by n features.
            y_true (ndarray Shape (m,)): The true labels.
            y_pred (ndarray Shape (m,)): The predicted labels.
        
        Returns:
            gradients_w (ndarray Shape (n,)): The gradient of the loss with respect to the weights.
            gradient_b (float): The gradient of the loss with respect to the bias.
        """

        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(X.transpose(), difference) + self.lambda_ * self.weights
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b
    
    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.lr * error_w
        self.bias = self.bias - self.lr * error_b
        
    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.epochs):
            x_dot_weights = np.matmul(self.weights, X.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_cost(y, pred)
            error_w, error_b = self.compute_gradients(X, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)
            
            # Print cost at every 10th times interval or as many iterations if < 10
            if i% math.ceil(self.epochs/10) == 0 or i == (self.epochs-1):
                print(f"Iteration {i:4}: Cost {float(loss):8.2f}   ")
            
    def predict(self, X):
        x_dot_weights = np.matmul(X, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]
    
    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    
def main():
    X, y = load_data("data/processed/dataset_1.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    my_lr = MyLogisticRegression(0.01, 1000, 0.005)
    my_lr.fit(X_train, y_train)
    y_pred = my_lr.predict(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)
    
    print("My logistic regression accuracy: ", accuracy_score(y_test, y_pred))
    print("Sckit-learn logistic regression accuracy: ", accuracy_score(y_test, y_pred_sklearn))
    
if __name__ == "__main__":
    main()