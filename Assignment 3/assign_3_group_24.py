import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Function to load data from sDAT and sNC files
def load_data(sDAT_file_path, sNC_file_path):
    sDAT_data = pd.read_csv(sDAT_file_path, header=None)
    sNC_data = pd.read_csv(sNC_file_path, header=None)

    sDAT_labels = np.ones(sDAT_data.shape[0])
    sNC_labels = np.zeros(sNC_data.shape[0])

    data = np.vstack((sDAT_data, sNC_data))
    labels = np.hstack((sDAT_labels, sNC_labels))

    return data, labels


# Function to tune SVM hyperparameters using GridSearchCV
def tune_SVM(train_data, train_labels, kernel, params):
    svm = SVC(kernel=kernel)
    grid_search = GridSearchCV(svm, params, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1, verbose=2)
    grid_search.fit(train_data, train_labels)

    return grid_search


# Function to plot performance of the SVM models
def plot_performance(grid_search, kernel):
    plt.figure()

    # Plotting performance for linear kernel
    if kernel == 'linear':
        plt.semilogx(grid_search.param_grid['C'], grid_search.cv_results_['mean_test_score'], label='Mean Test Score')
        plt.semilogx(grid_search.param_grid['C'], grid_search.cv_results_['mean_train_score'], label='Mean Train Score')
        plt.xlabel('C')
    # Plotting performance for polynomial kernel
    elif kernel == 'poly':
        mean_test_score = grid_search.cv_results_['mean_test_score'].reshape(len(grid_search.param_grid['C']),
                                                                             len(grid_search.param_grid['degree']))
        mean_train_score = grid_search.cv_results_['mean_train_score'].reshape(len(grid_search.param_grid['C']),
                                                                               len(grid_search.param_grid['degree']))
        for idx, degree in enumerate(grid_search.param_grid['degree']):
            plt.semilogx(grid_search.param_grid['C'], mean_test_score[:, idx], label=f'Test Score (degree {degree})')
            plt.semilogx(grid_search.param_grid['C'], mean_train_score[:, idx], label=f'Train Score (degree {degree})')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
    # Plotting performance for RBF kernel
    elif kernel == 'rbf':
        mean_test_score = grid_search.cv_results_['mean_test_score'].reshape(len(grid_search.param_grid['C']),
                                                                             len(grid_search.param_grid['gamma']))
        mean_train_score = grid_search.cv_results_['mean_train_score'].reshape(len(grid_search.param_grid['C']),
                                                                               len(grid_search.param_grid['gamma']))
        for idx, gamma in enumerate(grid_search.param_grid['gamma']):
            plt.semilogx(grid_search.param_grid['C'], mean_test_score[:, idx], label=f'Test Score (gamma {gamma})')
            plt.semilogx(grid_search.param_grid['C'], mean_train_score[:, idx], label=f'Train Score (gamma {gamma})')
        plt.xlabel('C')
        plt.ylabel('Accuracy')

    plt.legend()
    plt.title(f'{kernel.capitalize()} SVM Performance')
    plt.show()


# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['sNC', 'sDAT'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def Q1_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path):
    train_data, train_labels = load_data(train_sDAT_file_path, train_sNC_file_path)
    test_data, test_labels = load_data(test_sDAT_file_path, test_sNC_file_path)

    params = {'C': np.logspace(-3, 3, 7)}
    grid_search = tune_SVM(train_data, train_labels, 'linear', params)

    print(f"Best C for linear SVM: {grid_search.best_params_['C']}")

    linear_svm = grid_search.best_estimator_
    predictions = linear_svm.predict(test_data)
    print("Linear SVM performance:")
    print(classification_report(test_labels, predictions))

    plot_performance(grid_search, 'linear')
    plot_confusion_matrix(test_labels, predictions, 'Confusion Matrix for Linear SVM')


def Q2_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path):
    train_data, train_labels = load_data(train_sDAT_file_path, train_sNC_file_path)
    test_data, test_labels = load_data(test_sDAT_file_path, test_sNC_file_path)

    params = {'C': np.logspace(-3, 3, 7), 'degree': [2, 3, 4, 5]}
    grid_search = tune_SVM(train_data, train_labels, 'poly', params)

    print(f"Best C and degree for polynomial SVM: {grid_search.best_params_}")

    poly_svm = grid_search.best_estimator_
    predictions = poly_svm.predict(test_data)
    print("Polynomial SVM performance:")
    print(classification_report(test_labels, predictions))
    plot_performance(grid_search, 'poly')


def Q3_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path):
    train_data, train_labels = load_data(train_sDAT_file_path, train_sNC_file_path)
    test_data, test_labels = load_data(test_sDAT_file_path, test_sNC_file_path)

    params = {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
    grid_search = tune_SVM(train_data, train_labels, 'rbf', params)
    print(f"Best C and gamma for RBF SVM: {grid_search.best_params_}")

    rbf_svm = grid_search.best_estimator_
    predictions = rbf_svm.predict(test_data)
    print("RBF SVM performance:")
    print(classification_report(test_labels, predictions))
    plot_performance(grid_search, 'rbf')


if __name__ == "__main__":
    # Loading the data
    train_sDAT_file_path = "train.fdg_pet.sDAT.csv"
    train_sNC_file_path = "train.fdg_pet.sNC.csv"
    test_sDAT_file_path = "test.fdg_pet.sDAT.csv"
    test_sNC_file_path = "test.fdg_pet.sNC.csv"

    Q1_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path)
    Q2_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path)
    Q3_results(train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, test_sNC_file_path)
