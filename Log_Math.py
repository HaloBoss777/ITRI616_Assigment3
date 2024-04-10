# Library used to for multiclassification in logistical Regression
import numpy as np


# Defines a new class called MulticlassLogisticRegression


class MulticlassLogisticRegression:

    """
    constructor initializes the logistic regression model with a specified learning rate and number of iterations.
    It also initializes the weights and bias to None, which will be set later during the training process
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    """
    The softmax function is used to convert the raw output scores (logits) into probabilities for each class.
    It ensures that the output probabilities sum up to 1 for each sample
    """

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    """
    The fit method is used to train the logistic regression model on the training data X and target labels y
    """

    def fit(self, X, y):
        # calculates the number of samples, features, and unique classes in the dataset
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize weights and bias
        # Weights and Bias are initialized with zeros. The weights matrix has a shape of (n_features, n_classes) and the bias is a 1D array with length n_classes
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        # Gradient descent
        for _ in range(self.iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias

            # Softmax output
            y_pred = self.softmax(linear_model)

            # Compute gradients
            # Create an array to store gradients for weights
            dw = np.zeros_like(self.weights)
            # Iterate over each class to compute its gradient
            for k in range(n_classes):
                # Create a binary array where only the samples of class k are 1
                y_k = (y == k).astype(int)
                # Compute the gradient for class k
                dw[:, k] = np.dot(X.T, (y_pred[:, k] - y_k)) / n_samples
            # Compute the gradient for bias
            db = np.sum(y_pred - np.eye(n_classes)[y], axis=0) / n_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    """
    The method is used to make predictions on new input data X using the trained logistic regression model
    """

    def predict(self, X):
        # This line computes the linear part of the model
        # It multiplies the input data X with the model's weights (self.weights) and adds the bias (self.bias)
        linear_model = np.dot(X, self.weights) + self.bias
        # The raw output scores are passed through the softmax function, The softmax function converts the logits into probabilities for each class
        y_pred = self.softmax(linear_model)
        # The method returns the predicted class for each sample
        return np.argmax(y_pred, axis=1)

    """
    The method is used to evaluate the accuracy of the logistic regression model on a given dataset
    """

    def score(self, X, y):
        # This line uses the predict method (defined earlier in the class) to make predictions on the input data X
        # The result, y_pred, is an array of predicted class labels for each sample in X
        y_pred = self.predict(X)

        """
        This line calculates the accuracy of the model.
        It compares the predicted labels (y_pred) with the true labels (y) using the == operator,
        which produces an array of boolean values (True for correct predictions and False for incorrect ones)
        The np.mean function is then used to compute the mean of this boolean array, which gives the proportion of correct predictions 
        (i.e., the accuracy score). The accuracy score is a value between 0 and 1, where 1 indicates perfect accuracy.
        """
        return np.mean(y_pred == y)
