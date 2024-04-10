# import Pandas to structure our data for use
import pandas as pd

# MATH!!!!!
import numpy as np

# Dynamic file to choose
from tkinter import Tk
from tkinter.filedialog import askopenfile as openfile

# Used for Spliting data
from sklearn.model_selection import train_test_split

# Import Math Library created for logistical Regression
import Log_Math as logMath

# Label Encoding to turn categorical values in numerical values
from sklearn.preprocessing import LabelEncoder

# Import KNN math
import KNN_Math as KNN

# Import Functions to calculate Persition and F1 Score
from sklearn.metrics import precision_score, f1_score

# Import Plots
import matplotlib.pyplot as plt

# Find File to use


def loadfileCSV():

    # Create a Tk root window
    root = Tk()
    # Hide tkinter window
    root.withdraw()

    # Get CSV filepath
    filepath = openfile(filetype=[("CSV files", "*.csv")])

    # Check if file was selected
    if filepath:

        # Return filepath if a file was chosen
        return filepath

    else:

        # Retun nothing if no file was chosen
        print("No File Was Selected.")
        return None

# Function for cleaning CSV file and Prepare the file for use in the Logistic regression and K-nearest neighbour


def PrepareCSV_Penguin(filepath="penguin.csv"):
    # Load CSV file
    fileCSV = pd.read_csv(filepath)

    # # Print to show chosen data
    # print("\n" + '_' * 35 + 'First Rows of Dataset' + '_' * 35 + "\n")
    # print(fileCSV.head())

    # # Print summary of the CSV, including datatypes and non-null values for each column
    # print("\n" + '_' * 35 + 'Information of Dataset' + '_' * 35 + "\n")
    # print(fileCSV.info())
    # print("\n")

    # Remove Rows with null values
    fileCSV_Cleaned = fileCSV.dropna()

    # Drop uneeded columns (For penguin dataset = island, body_mass_g, sex, and year)
    fileCSV_Cleaned = fileCSV_Cleaned.drop(
        columns=["island", "body_mass_g", "sex", "year"])

    # Define features and target variable
    FeaturesCSV = fileCSV_Cleaned[[
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm"]]
    TargetCSV = fileCSV_Cleaned["species"]

    # Encode the target variable (0: Adelie, 1: Chinstrap, 2: Gentoo)
    label_encoder = LabelEncoder()
    EncodedTargetCSV = label_encoder.fit_transform(TargetCSV)

    # Split the data into training and testing sets
    X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data = train_test_split(
        FeaturesCSV, EncodedTargetCSV, test_size=0.3, random_state=42)

    # Convert the training and testing sets into numpy arrays
    # This array contains the training data features, which are the bill length, bill depth, and flipper length measurements for the penguins
    X_Train_Data = np.array(X_Train_Data)
    # This array contains the testing data features, similar to X_Train_Data but for the testing set.
    X_Test_Data = np.array(X_Test_Data)
    # This array contains the encoded target labels for the training set
    Y_Train_Data = np.array(Y_Train_Data)
    # This array contains the encoded target labels for the testing set, similar to Y_Train_Data but for the testing set.
    Y_Test_Data = np.array(Y_Test_Data)

    print("\n" + '_' * 35 + 'Shapes and First Rows of data' + '_' * 35 + "\n")
    print(f"X_Train_Data {X_Train_Data.shape}:\n{X_Train_Data[:5]}")
    print(f"\nX_Test_Data {X_Test_Data.shape}:\n{X_Test_Data[:5]}")
    print(f"\nY_Train_Data {Y_Train_Data.shape}:\n{Y_Train_Data[:5]}")
    print(f"\nY_Test_Data {Y_Test_Data.shape}:\n{Y_Test_Data[:5]}")

    return X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data


def main():
    # Get file to use in assignment
    filepath = loadfileCSV()

    # Get training and testing data (file into function to remove null values and unused columns)
    X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data = PrepareCSV_Penguin(
        filepath)

    # Logistic Regression model Before Hyperparameter---------------------------------------------------------------------------------------------------------------------

    Normal_lr_model = logMath.MulticlassLogisticRegression(
        learning_rate=0.01, iterations=1000)

    # Trains the logistic regression model
    Normal_lr_model.fit(X_Train_Data, Y_Train_Data)
    # Uses the trained logistic regression model to make predictions on the test data
    lr_predictions = Normal_lr_model.predict(X_Test_Data)
    # Calculates the accuracy of the logistic regression model
    lr_accuracy = Normal_lr_model.score(X_Test_Data, Y_Test_Data)
    # Calculates the precision of the logistic regression model using the precision_score function from sklearn.metrics
    lr_precision = precision_score(
        Y_Test_Data, lr_predictions, average='weighted')
    # Calculates the F1 score of the logistic regression model using the f1_score function from sklearn.metrics
    lr_f1_score = f1_score(Y_Test_Data, lr_predictions, average='weighted')
    # Print Result
    print(f"\n" + '_' * 35 +
          f"Normal Logistic Regression Learning Rate: 0.01" + '_' * 35 + "\n")
    print(f"Logistic Regression Accuracy: {lr_accuracy*100:.5f}%")
    print(f"Logistic Regression Precision: {lr_precision*100:.5f}%")
    print(f"Logistic Regression F1 Score: {lr_f1_score*100:.5f}%")

    # Hyperparameter tuning for Logistic Regression-----------------------------------------------------------------------------------------------------------------------
    learning_rates = [0.000001, 0.00001, 0.00007,
                      0.0001, 0.007, 0.0068, 0.005, 0.001, 0.01]
    best_lr_accuracy = 0
    best_lr = None

    # Works through each learning rate and finds the one with best accuracy
    for lr in learning_rates:
        model = logMath.MulticlassLogisticRegression(
            learning_rate=lr, iterations=1000)
        model.fit(X_Train_Data, Y_Train_Data)
        accuracy = model.score(X_Test_Data, Y_Test_Data)
        if accuracy > best_lr_accuracy:
            best_lr_accuracy = accuracy
            best_lr = lr

    # Train and evaluate the best Logistic Regression model
    """
    Creates an instance of the MulticlassLogisticRegression class from the logMath module,
    using the best learning rate (best_lr) found during hyperparameter tuning
    """
    best_lr_model = logMath.MulticlassLogisticRegression(
        learning_rate=best_lr, iterations=1000)

    # Trains the logistic regression model
    best_lr_model.fit(X_Train_Data, Y_Train_Data)
    # Uses the trained logistic regression model to make predictions on the test data
    lr_predictions = best_lr_model.predict(X_Test_Data)
    # Calculates the accuracy of the logistic regression model
    lr_accuracy = best_lr_model.score(X_Test_Data, Y_Test_Data)
    # Calculates the precision of the logistic regression model using the precision_score function from sklearn.metrics
    lr_precision = precision_score(
        Y_Test_Data, lr_predictions, average='weighted')
    # Calculates the F1 score of the logistic regression model using the f1_score function from sklearn.metrics
    lr_f1_score = f1_score(Y_Test_Data, lr_predictions, average='weighted')
    # Print Result
    print(f"\n" + '_' * 35 +
          f"Best Logistic Regression Learning Rate: {best_lr}" + '_' * 35 + "\n")
    print(f"Logistic Regression Accuracy: {lr_accuracy*100:.5f}%")
    print(f"Logistic Regression Precision: {lr_precision*100:.5f}%")
    print(f"Logistic Regression F1 Score: {lr_f1_score*100:.5f}%")

    # KNN model Before Hyperparameter---------------------------------------------------------------------------------------------------------------------

    # Uses a list comprehension to make predictions for each point in the test dataset (X_Test_Data) using the K-nearest neighbors (KNN) algorithm
    knn_predictions = [KNN.predict_classification(
        X_Train_Data, Y_Train_Data, test_point, 4) for test_point in X_Test_Data]
    # Calculates the accuracy of the KNN model by comparing the predicted class labels (best_knn_predictions) with the actual class labels in the test data (Y_Test_Data)
    knn_accuracy = np.mean(knn_predictions == Y_Test_Data)
    # Calculates the precision of the KNN model using the precision_score function from sklearn.metrics
    knn_precision = precision_score(
        Y_Test_Data, knn_predictions, average='weighted')
    # Calculates the F1 score of the KNN model using the f1_score function from sklearn.metrics
    knn_f1_score = f1_score(Y_Test_Data, knn_predictions, average='weighted')
    # Print Result
    print(f"\n" + '_' * 35 + f"Normal KNN k: 4" + '_' * 35 + "\n")
    print(f"KNN Accuracy: {knn_accuracy*100:.5f}%")
    print(f"KNN Precision: {knn_precision*100:.5f}%")
    print(f"KNN F1 Score: {knn_f1_score*100:.5f}%")

    # Hyperparameter tuning for KNN-----------------------------------------------------------------------------------------------------------------------
    k_values = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    best_k_accuracy = 0
    best_k = None

    # Works through each K value to find most accurate
    for k in k_values:
        knn_predictions = [KNN.predict_classification(
            X_Train_Data, Y_Train_Data, test_point, k) for test_point in X_Test_Data]
        accuracy = np.mean(knn_predictions == Y_Test_Data)
        if accuracy > best_k_accuracy:
            best_k_accuracy = accuracy
            best_k = k

    # Evaluate the best KNN model
    # Uses a list comprehension to make predictions for each point in the test dataset (X_Test_Data) using the K-nearest neighbors (KNN) algorithm
    best_knn_predictions = [KNN.predict_classification(
        X_Train_Data, Y_Train_Data, test_point, best_k) for test_point in X_Test_Data]
    # Calculates the accuracy of the KNN model by comparing the predicted class labels (best_knn_predictions) with the actual class labels in the test data (Y_Test_Data)
    knn_accuracy = np.mean(best_knn_predictions == Y_Test_Data)
    # Calculates the precision of the KNN model using the precision_score function from sklearn.metrics
    knn_precision = precision_score(
        Y_Test_Data, best_knn_predictions, average='weighted')
    # Calculates the F1 score of the KNN model using the f1_score function from sklearn.metrics
    knn_f1_score = f1_score(
        Y_Test_Data, best_knn_predictions, average='weighted')
    # Print Result
    print(f"\n" + '_' * 35 + f"Best KNN k: {best_k}" + '_' * 35 + "\n")
    print(f"KNN Accuracy: {knn_accuracy*100:.5f}%")
    print(f"KNN Precision: {knn_precision*100:.5f}%")
    print(f"KNN F1 Score: {knn_f1_score*100:.5f}%")


if __name__ == "__main__":
    main()
