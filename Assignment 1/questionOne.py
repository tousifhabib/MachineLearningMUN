import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_sNC = pd.read_csv("train.sNC.csv", header=None, delimiter=",")
train_sDAT = pd.read_csv("train.sDAT.csv", header=None, delimiter=",")

# Load the test data
test_sNC = pd.read_csv("test.sNC.csv", header=None, delimiter=",")
test_sDAT = pd.read_csv("test.sDAT.csv", header=None, delimiter=",")

# Load the 2D grid points
grid_points = pd.read_csv("2D_grid_points.csv", header=None, delimiter=",")
grid_points.columns = ["X", "Y"]


# Concatenate the training data and test data
train_data = pd.concat([train_sNC, train_sDAT], axis=0)
train_labels = np.concatenate((np.zeros(len(train_sNC)), np.ones(len(train_sDAT))))

test_data = pd.concat([test_sNC, test_sDAT], axis=0)
test_labels = np.concatenate((np.zeros(len(test_sNC)), np.ones(len(test_sDAT))))


k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
train_errors = []
test_errors = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Train the classifier
    knn.fit(train_data, train_labels)

    train_preds = knn.predict(train_data)
    test_preds = knn.predict(test_data)

    # Calculate the accuracy on the training and test data
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)

    train_errors.append(1 - train_accuracy)
    test_errors.append(1 - test_accuracy)

    print("K Value: ", k)
    print("Train accuracy")
    print(train_accuracy)
    print("Test accuracy")
    print(test_accuracy)

    # Plot the decision boundary and the data points
    X = grid_points.iloc[:, 0].values.reshape(-1, 1)
    Y = grid_points.iloc[:, 1].values.reshape(-1, 1)
    Z = knn.predict(np.c_[X, Y])

    plt.scatter(X, Y, c=Z)
    # plt.scatter(train_sNC[0], train_sNC[1], c='green', marker='+', label='sNC (Train)')
    # plt.scatter(train_sDAT[0], train_sDAT[1], c='blue', marker='+', label='sDAT (Train)')
    plt.scatter(test_sNC[0], test_sNC[1], c='red', marker='x', label='sNC (Test)')
    plt.scatter(test_sDAT[0], test_sDAT[1], c='orange', marker='x', label='sDAT (Test)')
    plt.title(k)
    plt.show()

plt.plot(k_values, train_errors, label='Train Error')
plt.plot(k_values, test_errors, label='Test Error')
plt.xlabel('K Value')
plt.ylabel('Error rate')
plt.show()
