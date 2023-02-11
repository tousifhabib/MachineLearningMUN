import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_sNC = pd.read_csv("train.sNC.csv")
train_sDAT = pd.read_csv("train.sDAT.csv")

# Load the test data
test_sNC = pd.read_csv("test.sNC.csv")
test_sDAT = pd.read_csv("test.sDAT.csv")

# Load the 2D grid points
grid_points = pd.read_csv("2D_grid_points.csv")

# Concatenate the training data and test data
train_data = pd.concat([train_sNC, train_sDAT], axis=0)
test_data = pd.concat([test_sNC, test_sDAT], axis=0)

# Assign class labels to the training data and test data
train_data["class_label"] = np.concatenate([np.zeros(236), np.ones(236)], axis=0)
test_data["class_label"] = np.concatenate([np.zeros(99), np.ones(99)], axis=0)

# Split the data into features and labels
train_features = train_data.iloc[:, :-1]
train_labels = train_data.iloc[:, -1]
test_features = test_data.iloc[:, :-1]
test_labels = test_data.iloc[:, -1]

# Train and evaluate the KNN classifier for various values of k
k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    train_preds = knn.predict(train_features)
    test_preds = knn.predict(test_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    
    # Plot the classification boundary and data samples
    x_min, x_max = grid_points["x"].min() - 0.1, grid_points["x"].max() + 0.1
    y_min, y_max = grid_points["y"].min() - 0.1, grid_points["y"].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points_features = np.c_[xx.ravel(), yy.ravel()]
    grid_points_preds = knn.predict(grid_points_features)
    grid_points_preds = grid_points_preds.resh
