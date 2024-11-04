import numpy as np
import pandas as pd


class LogisticRegressor():
    def __init__(self, learning_rate, iterations):  
        self.learning_rate = learning_rate         
        self.iterations = iterations 

    def fit(self, X, Y):
        if len(np.unique(Y)) != 2:
            raise ValueError("Cannot train on a single-class dataset.")
    
        self.m, self.n = X.shape  # num of rows = m, num of columns = n
        self.w = np.zeros(self.n)  # initializes the weights to a 0 vector of size n
        self.b = 0  # initializes bias to 0
        self.X = X         
        self.Y = Y 

        for i in range(self.iterations):             
            self.update_weights()   

    def update_weights(self):
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))  # sigmoid function

        # derivatives
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)

        # updating the weights & bias using gradient descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred
