import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

import utils

class Logreg:
    """
    A class to represent a Logistic Regression model with Lasso penalty (L1 regularization)
    for text classification.

    Attributes:
    -----------
    model : LogisticRegression
        The logistic regression model with Lasso penalty.
    selected_features_indices : list
        The indices of the selected features.

    Methods:
    --------
    train(X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5)
        Trains the Logistic Regression model using cross-validation to find the best C.
    predict(X)
        Predicts the class labels for the input data.
    get_top_k_features(vectorizer, k=5)
        Returns the top k features for each class.
    get_bottom_k_features(vectorizer, k=5)
        Returns the least k features for each class.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the LogisticRegressionLassoModel object.
        """
        self.model = None
        self.selected_features_indices = None

    def train(self, X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5, k=100):
        """
        Trains the Logistic Regression model using cross-validation to find the best C.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        Cs : list, optional
            The list of C values to try. Default is [0.01, 0.1, 1.0, 10].
        cv : int, optional
            The number of cross-validation folds. Default is 5.
        k : int, optional
            The number of features to use. Default is 100.

        Returns:
        --------
        None
        """
        if k == X_train.shape[1]:
            X_train_post_feature_selec = X_train
            self.selected_features_indices = range(X_train_post_feature_selec.shape[1])
        else:
            X_train_post_feature_selec, self.selected_features_indices = utils.get_top_features(X_train, y_train, k)

        param_grid = {'C': Cs}  # Cross-validate over different C values
        grid = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear'), param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_post_feature_selec, y_train)

        self.model = grid.best_estimator_
        print(f"Best C: {self.model.C}")

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        if not self.model:
            raise Exception("Model is not initialized. Please call the appropriate method to initialize the model.")

        X_post_feature_selec = X[:, self.selected_features_indices]
        y_pred = self.model.predict(X_post_feature_selec)
        return y_pred

    def get_top_k_features(self, vectorizer, k=5):
        """
        Returns the top k features for each class.

        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            The number of top features to return. Default is 5.

        Returns:
        --------
        top_k_features : dict
            A dictionary containing the top k features for each class.
        """
        feature_importance = np.abs(self.model.coef_[0])  # Absolute values of coefficients
        top_k_indices = np.argsort(feature_importance)[-k:]  # Indices of top k features
        feature_names = np.array(vectorizer.get_feature_names_out())[self.selected_features_indices]

        print(f"Top {k} features for classification: {feature_names[top_k_indices]}")
        return feature_names[top_k_indices]

    def get_bottom_k_features(self, vectorizer, k=5):
        """
        Returns the least k features for each class.

        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            The number of bottom features to return. Default is 5.

        Returns:
        --------
        bottom_k_features : array
            An array containing the bottom k features for the class.
        """
        feature_importance = np.abs(self.model.coef_[0])
        bottom_k_indices = np.argsort(feature_importance)[:k]  # Indices of bottom k features
        feature_names = np.array(vectorizer.get_feature_names_out())[self.selected_features_indices]

        print(f"Bottom {k} features for classification: {feature_names[bottom_k_indices]}")
        return feature_names[bottom_k_indices]


def run_the_logistic_model(dataset_name, X_train, y_train, X_test, y_test, vectorizer):
    """
        Run the logistic regression model with Lasso penalty on the given train and test sets.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset (unigrams or bigrams).
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.
        y_test : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        df_evaluation : dataframe
            A dataframe with the evaluation scores and info for further analysis.
    """
    evaluations = []

    total_count_features = X_train.shape[1]
    number_feature_range = utils.get_feature_range(total_count_features)

    for k in number_feature_range:
        print(f"Running with {k} features")
        model = Logreg()
        model.train(X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5, k=k)
        y_pred = model.predict(X_test)

        df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)

        df_top_5 = model.get_top_k_features(vectorizer, k=5)
        df_bottom_5 = model.get_bottom_k_features(vectorizer, k=5)

        new_row = {
            'model_name': f'Logistic Regression with Lasso (#{k} features)',
            'dataset_name': dataset_name,
            **df_scores,
            'top_5_features_deceptive': ", ".join(df_top_5['deceptive']),
            'top_5_features_truthful': ", ".join(df_top_5['truthful']),
            'bottom_5_features_deceptive': ", ".join(df_bottom_5['deceptive']),
            'bottom_5_features_truthful': ", ".join(df_bottom_5['truthful'])
        }

        evaluations.append(new_row)

    df_evaluations = pd.DataFrame(evaluations)
    return df_evaluations, y_pred
