# Used for numerical operations
import numpy as np


# Used to calculate the distance between points
def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between point1 and point2. 
    It converts the points into NumPy arrays, subtracts one from the other, squares the result, 
    sums up the squared differences, and takes the square root of the sum
    """
    distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    # Returnes Result
    return distance


# Takes the training data, a test point, and a value k as arguments


def get_neighbors(X_train, y_train, test_point, k):
    # Initializes an empty list to store the distances between the test point and each training point
    distances = []
    # Iterates over the training data points.
    for index, train_point in enumerate(X_train):
        # Calculates the Euclidean distance between the current training point and the test point
        distance = euclidean_distance(train_point, test_point)
        # Appends a tuple containing the training point, its label, and its distance from the test point to the distances list
        distances.append((train_point, y_train[index], distance))

    # Sorts the list of distances in ascending order based on the distance
    distances.sort(key=lambda x: x[2])
    # Selects the first k elements from the sorted list, which are the nearest neighbors
    neighbors = distances[:k]
    # Returns the list of nearest neighbors
    return neighbors


# predict the classification for a test point based on its k nearest neighbors


def predict_classification(X_train, y_train, test_point, k):

    # Retrieves the k nearest neighbors of the test point
    neighbors = get_neighbors(X_train, y_train, test_point, k)

    # Initializes a dictionary to count votes for each class.
    class_votes = {}

    # Loops through the nearest neighbors
    for neighbor in neighbors:

        # Extracts the label of the neighbor
        label = neighbor[1]

        # Increments the vote for the neighbor's class in class_votes. If the class is not yet in class_votes, it adds it with a vote of 1
        if label in class_votes:

            class_votes[label] += 1

        else:

            class_votes[label] = 1

        # Sorts the classes by vote count in descending order
        sorted_votes = sorted(class_votes.items(),
                              key=lambda x: x[1], reverse=True)

    # Returns the class with the highest vote count as the prediction
    return sorted_votes[0][0]
