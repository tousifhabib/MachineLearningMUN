import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def simple_linear_regression(train_df, test_df):
    X_train = train_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_train = train_df["ConcreteCompressiveStrength_MPa_Megapascals_"]
    X_test = test_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_test = test_df["ConcreteCompressiveStrength_MPa_Megapascals_"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # We choose 5-fold cross-validation because it is a commonly used value that provides a good balance
    # between computational cost and model performance estimation. Moreover, with the given dataset size,
    # it provides enough data for each fold to ensure a reasonable assessment of the model's performance.
    # (JAXON PLS LET ME KNOW IF THIS IS CORRECT AND TRY PLAYING AROUND WITH THE NUMBERS)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean_r2 = np.mean(cv_scores)

    return rse, r2, cv_mean_r2, y_test, y_pred


def ridge_regression(train_df, test_df):
    X_train = train_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_train = train_df["ConcreteCompressiveStrength_MPa_Megapascals_"]
    X_test = test_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_test = test_df["ConcreteCompressiveStrength_MPa_Megapascals_"]

    params = {'alpha': np.logspace(-4, 4, 100)}
    grid_search = GridSearchCV(Ridge(), params, cv=5, scoring='r2', return_train_score=True)
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return best_alpha, rse, r2, grid_search.cv_results_


def lasso_regression(train_df, test_df):
    X_train = train_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_train = train_df["ConcreteCompressiveStrength_MPa_Megapascals_"]
    X_test = test_df.drop(columns=["ConcreteCompressiveStrength_MPa_Megapascals_"])
    y_test = test_df["ConcreteCompressiveStrength_MPa_Megapascals_"]

    params = {'alpha': np.logspace(-4, 4, 100)}
    grid_search = GridSearchCV(Lasso(), params, cv=5, scoring='r2', return_train_score=True)
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return best_alpha, rse, r2, grid_search.cv_results_


def plot_alpha_search(cv_results, model_name):
    alphas = cv_results["param_alpha"]
    train_scores = cv_results["mean_train_score"]
    test_scores = cv_results["mean_test_score"]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, train_scores, label='Train')
    plt.plot(alphas, test_scores, label='Test')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('R^2 score')
    plt.title(f'{model_name} Model Performance')
    plt.legend()
    plt.show()

def plot_true_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True Compressive Strength')
    plt.ylabel('Predicted Compressive Strength')
    plt.title(f'{model_name} True vs. Predicted Compressive Strength')
    plt.show()

def Q1_results(train_df, test_df):
    rse, r2, cv_mean_r2, y_test, y_pred = simple_linear_regression(train_df, test_df)
    print(f"Simple Linear Regression Results:")
    print(f"RSE: {rse}")
    print(f"R^2: {r2}")
    print(f"Cross-Validation R^2: {cv_mean_r2}")
    plot_true_vs_predicted(y_test, y_pred, "Simple Linear Regression")

def Q2_results(train_df, test_df):
    best_alpha, rse, r2, cv_results = ridge_regression(train_df, test_df)
    print(f"Ridge Regression Results:")
    print(f"Best alpha: {best_alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {r2}")
    plot_alpha_search(cv_results, "Ridge Regression")

def Q3_results(train_df, test_df):
    best_alpha, rse, r2, cv_results = lasso_regression(train_df, test_df)
    print(f"Lasso Regression Results:")
    print(f"Best alpha: {best_alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {r2}")
    plot_alpha_search(cv_results, "Lasso Regression")

if __name__ == "__main__":
    train_file_path = "train.csv"
    test_file_path = "test.csv"

    train_df, test_df = load_data(train_file_path, test_file_path)

    Q1_results(train_df, test_df)
    Q2_results(train_df, test_df)
    Q3_results(train_df, test_df)
