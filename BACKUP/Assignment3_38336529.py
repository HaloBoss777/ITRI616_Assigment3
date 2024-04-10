#import Pandas to structure our data for use
import pandas as pd

#MATH!!!!!
import numpy as np

#Dynamic file to choose
from tkinter import Tk
from tkinter.filedialog import askopenfile as openfile

#Used for Spliting data
from sklearn.model_selection import train_test_split

#used for the multiclass logistical Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Find File to use
def loadfileCSV():

  # Create a Tk root window
  root = Tk()
  # Hide tkinter window
  root.withdraw()

  #Get CSV filepath
  filepath = openfile(filetype=[("CSV files", "*.csv")])

  #Check if file was selected
  if filepath:

    #Return filepath if a file was chosen
    return filepath

  else:

    #Retun nothing if no file was chosen
    print("No File Was Selected.")
    return None

#Function for cleaning CSV file and Prepare the file for use in the Logistic regression and K-nearest neighbour
def PrepareCSV(filepath="penquin.csv"):

  #Load CSV file
  fileCSV = pd.read_csv(filepath)

  #Print to show chosen data
  print("\n" + '_' * 35 + 'First Rows of Dataset' + '_' * 35 + "\n")
  print(fileCSV.head())

  #Print summary of the CSV, including datatypes and non-null values for each column
  print("\n" + '_' * 35 + 'Information of Dataset' + '_' * 35 + "\n")
  print(fileCSV.info())
  print("\n")

  #Remove Rows with null values
  fileCSV_Cleaned = fileCSV.dropna()

  #Drop uneeded columns (For penguin dataset = island, body_mass_g, sex, and year)
  fileCSV_Cleaned = fileCSV_Cleaned.drop(columns=["island", "body_mass_g", "sex", "year"])

  # Define features and target variable
  FeaturesCSV = fileCSV_Cleaned[["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]]
  TargetCSV = fileCSV_Cleaned["species"]

  # Split the data into training and testing sets
  X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data = train_test_split(FeaturesCSV, TargetCSV, test_size=0.3, random_state=42)

  # Convert the training and testing sets into numpy arrays
  X_Train_Data = np.array(X_Train_Data)
  X_Test_Data = np.array(X_Test_Data)
  Y_Train_Data = np.array(Y_Train_Data)
  Y_Test_Data = np.array(Y_Test_Data)

  #Print Data
  print(X_Train_Data.shape, X_Test_Data.shape, Y_Train_Data.shape, Y_Test_Data.shape)

  return X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data


def main():
  # Get file to use in assignment
  filepath = loadfileCSV()

  # Get training and testing data (file into function to remove null values and unused columns)
  X_Train_Data, X_Test_Data, Y_Train_Data, Y_Test_Data = PrepareCSV(filepath)

  # Train the model
  model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

  #Use the Obserations to predict the species of a penguin
  model.fit(X_Train_Data, Y_Train_Data)

  # Evaluate the model
  predictions = model.predict(X_Test_Data)
  accuracy = accuracy_score(Y_Test_Data, predictions)
  print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()