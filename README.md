# Iris Flower Classification using PyTorch

## Purpose

This code implements a neural network using PyTorch to classify iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The model is trained on the famous Iris dataset and aims to predict the species of an unknown iris flower based on its sepal and petal measurements.

## Approach

The code follows these steps:

1. **Data Loading and Preprocessing:**
   - Loads the Iris dataset from a CSV file using Pandas.
   - Removes the unnecessary 'Id' column.
   - Renames the columns for better readability.
   - Maps the species names to numerical labels (0, 1, 2).
   - Visualizes the data using scatter plots to understand the relationships between features.

2. **Data Splitting:**
   - Splits the dataset into training and testing sets using `train_test_split` from scikit-learn.
   - Converts the data into PyTorch tensors for compatibility with the neural network.

3. **Model Building:**
   - Defines a neural network model with two hidden layers using PyTorch's `nn.Module`.
   - Uses ReLU activation function for non-linearity.
   - The model takes 4 input features (sepal length, sepal width, petal length, petal width) and outputs 3 probabilities for each species.

4. **Training:**
   - Defines the loss function (CrossEntropyLoss) and optimizer (Adam).
   - Trains the model for a specified number of epochs, iteratively updating the model's weights to minimize the loss.
   - Prints the loss during training to monitor progress.
   - Plots the training loss over epochs to visualize the learning curve.

5. **Evaluation:**
   - Evaluates the model's performance on the testing set by predicting the species for each unknown iris.
   - Calculates the accuracy of the model by comparing the predictions with the true labels.
   - Displays the predictions and accuracy in a Pandas DataFrame.

6. **Prediction on Unknown Iris:**
   - Creates a PyTorch tensor representing the features of an unknown iris flower.
   - Uses the trained model to predict the species of the unknown iris.
   - Prints the predicted species label and probabilities.
   - Visualizes the unknown iris on the scatter plots to see its relative position to other irises.
