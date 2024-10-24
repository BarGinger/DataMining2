"""
File: Multinomial_Naive_Bayes.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Multinomial Naive Bayes model
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import utils


class MultinomialNaiveBayesModel:
    """
    A class to represent a Multinomial Naive Bayes model for text classification.

    Attributes:
    -----------
    model : MultinomialNB
        The Multinomial Naive Bayes model.
    method : str
        The method used for feature selection. Default is 'l1'.
    max_features : int
        The maximum number of features to be selected. Default is 100.
    selected_features_indices : list
        The indices of the selected features.

    Methods:
    --------
    train(X_train, y_train, alphas=[0.1, 0.5, 1.0], cv=5)
        Trains the Multinomial Naive Bayes model using cross-validation to find the best alpha.
    predict(X)
        Predicts the class labels for the input data.
    get_top_k_features(vectorizer, k=5)
        Returns the top k features for each class (deceptive and truthful).
    get_bottom_k_features(vectorizer, k=5)
        Returns the least k features for each class (deceptive and truthful).
    """

    def __init__(self, method='l1', max_features=100):
        """
        Constructs all the necessary attributes for the MultinomialNaiveBayesModel object.

        Parameters:
        -----------
        method : str
            The method used for feature selection. Default is 'l1'.
        max_features : int
            The maximum number of features to be selected. Default is 100.
        """
        self.model = None
        self.method = method
        self.max_features = max_features
        self.selected_features_indices = None

    def train(self, X_train, y_train, alphas=[0.1, 0.5, 1.0], cv=5):
        """
        Trains the Multinomial Naive Bayes model using cross-validation to find the best alpha.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        alphas : list, optional
            The list of alpha values to try. Default is [0.1, 0.5, 1.0].
        cv : int, optional
            The number of cross-validation folds. Default is 5.

        Returns:
        --------
        None
        """
        # Perform feature selection using the train data set
        if 'none' in self.method:
            X_train_post_feature_selec = X_train
            self.selected_features_indices = range(X_train_post_feature_selec.shape[1])
        else:
            X_train_post_feature_selec, self.selected_features_indices = utils.feature_selection(X_train,
                                                                                                 y_train,
                                                                                                 self.method,
                                                                                                 self.max_features)

        # Train the Multinomial Naive Bayes model using cross-validation to find the best alpha
        param_grid = {'alpha': alphas}  # Cross-validate over different alpha values
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train_post_feature_selec, y_train)

        # Save the best model after cross-validation
        self.model = grid.best_estimator_
        print(f"Best alpha: {self.model.alpha}")

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
        Returns the top k features for each class (deceptive and truthful).

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
        feature_log_probs = self.model.feature_log_prob_
        top_k_deceptive = np.argsort(feature_log_probs[1])[-k:]  # Top k for 'deceptive'
        top_k_truthful = np.argsort(feature_log_probs[0])[-k:]  # Top k for 'truthful'

        feature_names = np.array(vectorizer.get_feature_names_out())[self.selected_features_indices]
        print(f"Top {k} features for deceptive reviews: {feature_names[top_k_deceptive]}")
        print(f"Top {k} features for truthful reviews: {feature_names[top_k_truthful]}")

        return {
            'deceptive': feature_names[top_k_deceptive],
            'truthful': feature_names[top_k_truthful]
        }

    def get_bottom_k_features(self, vectorizer, k=5):
        """
        Returns the least k features for each class (deceptive and truthful).

        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            The number of bottom features to return. Default is 5.

        Returns:
        --------
        bottom_k_features : dict
            A dictionary containing the bottom k features for each class.
        """
        feature_log_probs = self.model.feature_log_prob_
        bottom_k_deceptive = np.argsort(feature_log_probs[1])[:k]  # Bottom k for 'deceptive'
        bottom_k_truthful = np.argsort(feature_log_probs[0])[:k]  # Bottom k for 'truthful'

        feature_names = np.array(vectorizer.get_feature_names_out())[self.selected_features_indices]
        print(f"Bottom {k} features for deceptive reviews: {feature_names[bottom_k_deceptive]}")
        print(f"Bottom {k} features for truthful reviews: {feature_names[bottom_k_truthful]}")

        return {
            'deceptive': feature_names[bottom_k_deceptive],
            'truthful': feature_names[bottom_k_truthful]
        }

def run_the_model(dataset_name, X_train, y_train, X_test, y_test, vectorizer):
    """
        Run the current model with the given train and test sets

        Parameters:
        -----------
        dataset_name : str
            The name of the given dataset (unigrams or bigrams)
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.
        y_test : array-like of shape (n_samples,)
            The target values of the test set.

        Returns:
        --------
        df_evaluation : dataframe
            A dataframe with the evaluation scores and needed info for further analysis
        """

    # Prepare a list to collect evaluation results
    evaluations = []

    for method in ['none', 'l1', 'l2']:
        print(f"Using feature selection: {method}")
        model = MultinomialNaiveBayesModel(method=method, max_features=100)
        model.train(X_train, y_train, alphas=[0.1, 0.15, 0,25, 0.5, 0.75, 0.85, 1.0], cv=5)
        y_pred = model.predict(X_test)

        # Calculate scores
        df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)

        # Get top and bottom 5 features
        df_top_5 = model.get_top_k_features(vectorizer, k=5)
        df_bottom_5 = model.get_bottom_k_features(vectorizer, k=5)

        # Create a new row for the DataFrame
        new_row = {
            'model_name': f'Multinomial Naive Bayes ({method})',
            'dataset_name': dataset_name,
            **df_scores,  # Unpack scores
            'top_5_features_deceptive': ", ".join(df_top_5['deceptive']),
            'top_5_features_truthful': ", ".join(df_top_5['truthful']),
            'bottom_5_features_deceptive': ", ".join(df_bottom_5['deceptive']),
            'bottom_5_features_truthful': ", ".join(df_bottom_5['truthful'])
        }

        # Append new_row to evaluations list
        evaluations.append(new_row)

    # Create DataFrame from the list of evaluations
    df_evaluations = pd.DataFrame(evaluations)
    return  df_evaluations