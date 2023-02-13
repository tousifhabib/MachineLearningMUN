import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path, header=None, delimiter=",")


def load_grid_points():
    grid_points = pd.read_csv("2D_grid_points.csv", header=None, delimiter=",")
    grid_points.columns = ["X", "Y"]
    return grid_points

def load_and_concat_data(train_path1, train_path2, test_path1, test_path2):
    train_data1 = load_data(train_path1)
    train_data2 = load_data(train_path2)
    test_data1 = load_data(test_path1)
    test_data2 = load_data(test_path2)

    train_data = pd.concat([train_data1, train_data2], axis=0)
    train_labels = np.concatenate((np.zeros(len(train_data1)), np.ones(len(train_data2))))

    test_data = pd.concat([test_data1, test_data2], axis=0)
    test_labels = np.concatenate((np.zeros(len(test_data1)), np.ones(len(test_data2))))

    return train_data, train_labels, test_data, test_labels

def train_classifier(metric, k, train_data, train_labels):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(train_data, train_labels)
    return knn


def predict_labels(classifier, data):
    return classifier.predict(data)


def calculate_errors(classifier, train_data, train_labels, test_data, test_labels):
    train_preds = predict_labels(classifier, train_data)
    test_preds = predict_labels(classifier, test_data)

    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)

    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy

    return train_error, test_error

def visualize_results(metric, k, train_errors, test_errors, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT,
                      classifier):
    grid_preds = predict_labels(classifier, grid_points)

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title(metric + "\n" "k = " + str(k) + "\n" + " (Train Error: " + str(round(train_errors[-1], 4)) +
              ", Test Error: " + str(round(test_errors[-1], 4)) + ")")
    plt.scatter(grid_points['X'], grid_points['Y'], c=grid_preds)
    plt.scatter(train_sNC[0], train_sNC[1], c='purple', marker='o', label='sNC (Train)')
    plt.scatter(train_sDAT[0], train_sDAT[1], c='orange', marker='x', label='sDAT (Train)')
    plt.scatter(test_sNC[0], test_sNC[1], c='green', marker='o', label='sNC (Test)')
    plt.scatter(test_sDAT[0], test_sDAT[1], c='blue', marker='x', label='sDAT (Test)')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def classify(metric, k_values, train_data, train_labels, test_data, test_labels, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT):
    train_errors = []
    test_errors = []

    for k in k_values:
        classifier = train_classifier(metric, k, train_data, train_labels)

        train_error, test_error = calculate_errors(classifier, train_data, train_labels, test_data, test_labels)
        train_errors.append(train_error)
        test_errors.append(test_error)

        visualize_results(metric, k, train_errors, test_errors, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT, classifier)

    return train_errors, test_errors


def generate_error_rate_curve(metric, k_values, train_data, train_labels, test_data, test_labels):
    train_errors = []
    test_errors = []
    model_capacity = [1 / k for k in k_values]

    for k in k_values:
        classifier = train_classifier(metric, k, train_data, train_labels)

        train_error, test_error = calculate_errors(classifier, train_data, train_labels, test_data, test_labels)
        train_errors.append(train_error)
        test_errors.append(test_error)

    return model_capacity, train_errors, test_errors


def Q1_results():
    print('Generating results for Q1...')
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]

    train_sNC = load_data(train_sNC_file_path)
    train_sDAT = load_data(train_sDAT_file_path)
    test_sNC = load_data(test_sNC_file_path)
    test_sDAT = load_data(test_sDAT_file_path)
    grid_points = load_grid_points()
    train_data, train_labels, test_data, test_labels = load_and_concat_data(train_sNC_file_path, train_sDAT_file_path,
                                                                            test_sNC_file_path, test_sDAT_file_path)

    train_errors, test_errors = classify('euclidean', k_values, train_data, train_labels, test_data, test_labels, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT)

    plt.title('Error rate for Euclidean')
    plt.plot(k_values, train_errors, label='Train Error')
    plt.plot(k_values, test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('K Value')
    plt.ylabel('Error rate')
    plt.show()


def Q2_results():
    print('Generating results for Q2...')
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]

    train_sNC = load_data(train_sNC_file_path)
    train_sDAT = load_data(train_sDAT_file_path)
    test_sNC = load_data(test_sNC_file_path)
    test_sDAT = load_data(test_sDAT_file_path)
    grid_points = load_grid_points()
    train_data, train_labels, test_data, test_labels = load_and_concat_data(train_sNC_file_path, train_sDAT_file_path,
                                                                            test_sNC_file_path, test_sDAT_file_path)

    train_errors, test_errors = classify('manhattan', k_values, train_data, train_labels, test_data, test_labels, grid_points, train_sNC, train_sDAT, test_sNC, test_sDAT)

    plt.title('Error rate for Manhattan')
    plt.plot(k_values, train_errors, label='Train Error')
    plt.plot(k_values, test_errors, label='Test Error')
    plt.legend()
    plt.xlabel('K Value')
    plt.ylabel('Error rate')
    plt.show()


def Q3_results():
    print('Generating results for Q3...')

    train_data, train_labels, test_data, test_labels = load_and_concat_data(train_sNC_file_path, train_sDAT_file_path,
                                                                            test_sNC_file_path, test_sDAT_file_path)

    euclidean_model_capacity, euclidean_train_errors, euclidean_test_errors = generate_error_rate_curve('euclidean', range(1, 201), train_data, train_labels, test_data, test_labels)
    manhattan_model_capacity, manhattan_train_errors, manhattan_test_errors = generate_error_rate_curve('manhattan', range(1, 201), train_data, train_labels, test_data, test_labels)

    plt.semilogx(euclidean_model_capacity, euclidean_train_errors, label='Euclidean Train Error')
    plt.semilogx(euclidean_model_capacity, euclidean_test_errors, label='Euclidean Test Error')
    plt.semilogx(manhattan_model_capacity, manhattan_train_errors, label='Manhattan Train Error')
    plt.semilogx(manhattan_model_capacity, manhattan_test_errors, label='Manhattan Test Error')
    plt.xlabel('Model Capacity (1/k)')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()



if __name__ == "__main__":

    train_sNC_file_path = "train.sNC.csv"
    train_sDAT_file_path = "train.sDAT.csv"
    test_sNC_file_path = "test.sNC.csv"
    test_sDAT_file_path = "test.sDAT.csv"

    Q1_results()
    Q2_results()
    Q3_results()
