import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Global variables
train_sNC_file_path = "train.sNC.csv"
train_sDAT_file_path = "train.sDAT.csv"
test_sNC_file_path = "test.sNC.csv"
test_sDAT_file_path = "test.sDAT.csv"

def classify(metric, k_values):
    # Load the training data
    train_sNC = pd.read_csv(train_sNC_file_path, header=None, delimiter=",")
    train_sDAT = pd.read_csv(train_sDAT_file_path, header=None, delimiter=",")

    # Load the test data
    test_sNC = pd.read_csv(test_sNC_file_path, header=None, delimiter=",")
    test_sDAT = pd.read_csv(test_sDAT_file_path, header=None, delimiter=",")

    # Load the 2D grid points
    grid_points = pd.read_csv("2D_grid_points.csv", header=None, delimiter=",")
    grid_points.columns = ["X", "Y"]

    # Concatenate the training data and test data
    train_data = pd.concat([train_sNC, train_sDAT], axis=0)
    train_labels = np.concatenate((np.zeros(len(train_sNC)), np.ones(len(train_sDAT))))

    test_data = pd.concat([test_sNC, test_sDAT], axis=0)
    test_labels = np.concatenate((np.zeros(len(test_sNC)), np.ones(len(test_sDAT))))

    train_errors = []
    test_errors = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        # Train the classifier
        knn.fit(train_data, train_labels)

        train_preds = knn.predict(train_data)
        test_preds = knn.predict(test_data)

        # Calculate the accuracy on the training and test data
        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        train_errors.append(1 - train_accuracy)
        test_errors.append(1 - test_accuracy)

        # Predict the labels for the 2D grid points
        grid_preds = knn.predict(grid_points)

        # Plot the decision boundary using a contour plot
        plt.figure(figsize=(8, 8), dpi=80)
        plt.title(metric + "\n" "k = " + str(k) + "\n" + " (Train Error: " + str(round(train_errors[-1], 4)) +
                  ", Test Error: " + str(round(test_errors[-1], 4)) + ")")
        plt.scatter(grid_points['X'], grid_points['Y'], c=grid_preds)
        # plt.scatter(train_sNC[0], train_sNC[1], c='purple', marker='o', label='sNC (Train)')
        # plt.scatter(train_sDAT[0], train_sDAT[1], c='orange', marker='x', label='sDAT (Train)')
        plt.scatter(test_sNC[0], test_sNC[1], c='green', marker='o', label='sNC (Test)')
        plt.scatter(test_sDAT[0], test_sDAT[1], c='blue', marker='x', label='sDAT (Test)')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return train_errors, test_errors

def Q1_results():
    print('Generating results for Q1...')
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    train_errors, test_errors = classify('euclidean', k_values)
    plt.title('Error rate for Euclidean')
    plt.plot(k_values, train_errors, label='Train Error')
    plt.plot(k_values, test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('K Value')
    plt.ylabel('Error rate')
    plt.show()

def Q2_results():
    print('Generating results for Q2...')
    # K value 30 should be utilized from question 1
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    train_errors, test_errors = classify('manhattan', k_values)
    plt.title('Error rate for Manhattan')
    plt.plot(k_values, train_errors, label='Train Error')
    plt.plot(k_values, test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('K Value')
    plt.ylabel('Error rate')
    plt.show()

def Q3_results():
    print('Generating results for Q3...')
    metric = 'manhattan'  # chosen distance metric from Q2 because of Occams Razor
    k_values = range(1, 201)
    train_errors = []
    test_errors = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        # Load the training data
        train_sNC = pd.read_csv(train_sNC_file_path, header=None, delimiter=",")
        train_sDAT = pd.read_csv(train_sDAT_file_path, header=None, delimiter=",")

        # Load the test data
        test_sNC = pd.read_csv(test_sNC_file_path, header=None, delimiter=",")
        test_sDAT = pd.read_csv(test_sDAT_file_path, header=None, delimiter=",")

        # Load the 2D grid points
        grid_points = pd.read_csv("2D_grid_points.csv", header=None, delimiter=",")
        grid_points.columns = ["X", "Y"]

        # Concatenate the training data and test data
        train_data = pd.concat([train_sNC, train_sDAT], axis=0)
        train_labels = np.concatenate((np.zeros(len(train_sNC)), np.ones(len(train_sDAT))))

        test_data = pd.concat([test_sNC, test_sDAT], axis=0)
        test_labels = np.concatenate((np.zeros(len(test_sNC)), np.ones(len(test_sDAT))))

        # Train the classifier
        knn.fit(train_data, train_labels)

        train_preds = knn.predict(train_data)
        test_preds = knn.predict(test_data)

        # Calculate the accuracy on the training and test data
        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        train_errors.append(1 - train_accuracy)
        test_errors.append(1 - test_accuracy)

    model_capacity = [1 / k for k in k_values]
    plt.semilogx(model_capacity, train_errors, label='Train Error')
    plt.semilogx(model_capacity, test_errors, label='Test Error')
    plt.xlabel('Model Capacity (1/k)')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()

