from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import pandas as pd
import utils


class Logreg:
    """
    Logistic Regression model with Lasso (L1) or Ridge (L2) regularization for text classification.

    Attributes:
    -----------
    model : LogisticRegression
        The logistic regression model with the specified penalty.
    penalty : str
        The regularization type ('l1' for Lasso or 'l2' for Ridge).
    selected_features_indices : list
        Indices of the selected features if feature selection is applied.

    Methods:
    --------
    train(X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5)
        Trains the Logistic Regression model using cross-validation to find the best C.
    predict(X)
        Predicts the class labels for the input data.
    get_top_k_features(vectorizer, k=5)
        Returns the top k most influential features.
    get_bottom_k_features(vectorizer, k=5)
        Returns the k least influential features.
    """

    def __init__(self, penalty='l1'):
        """
        Initializes the LogisticRegressionModel object.

        Parameters:
        -----------
        penalty : str, optional
            Regularization type, either 'l1' for Lasso or 'l2' for Ridge (default is 'l1').
        """
        if penalty not in ['l1', 'l2']:
            raise ValueError("Penalty must be either 'l1' or 'l2'.")

        self.model = None
        self.penalty = penalty

    def train(self, X_train, y_train, Cs=[0.0001, 0.001, 0.01, 0.1, 1.0, 10], cv=5):
        """
           Trains the Logistic Regression model using LogisticRegressionCV to find the best C
           and performs optional feature selection.

           Parameters:
           -----------
           X_train : array-like of shape (n_samples, n_features)
               Training data features.
           y_train : array-like of shape (n_samples,)
               Training data labels.
           Cs : list, optional
               List of C values for cross-validation (default is [0.001, 0.01, 0.1, 1.0, 10]).
           cv : int, optional
               Number of cross-validation folds (default is 5).
           min_coef_threshold : float, optional
               Minimum absolute coefficient threshold for feature selection (default is 0.1).

           Returns:
           --------
           The parameters of the best model
           """
        # Set up LogisticRegressionCV with penalty, Cs, and cross-validation
        self.model = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            penalty=self.penalty,
            solver='liblinear',
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit model and automatically select the best C
        self.model.fit(X_train, y_train)
        print(f"Optimal C: {self.model.C_[0]}")

        total_count = self.model.coef_.size
        non_zero_weights_count = (self.model.coef_ != 0).sum()
        zero_weights_count = total_count - non_zero_weights_count

        print(f"out of {total_count} there are {zero_weights_count} zero weights and {non_zero_weights_count} nonzero weights")

        return {
            'C':self.model.C_[0],
            'cv':cv,
            'penalty': self.penalty,
            'solver': 'liblinear',
            'scoring': 'accuracy'
        }

    def predict(self, X):
        """
        Predicts class labels for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        if not self.model:
            raise Exception("Model not initialized. Call the 'train' method first.")

        if hasattr(self, 'selected_features_mask'):
            X = X[:, self.selected_features_mask]

        y_pred = self.model.predict(X)
        return y_pred

    def get_top_k_features(self, vectorizer, model_name, dataset_name, k=5):
        """
        Returns the top k most influential features for each class in a binary classification model,
        with feature names and coefficients as separate entries.

        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        model_name : str
            Name of the current model

        dataset_name : str
            Name of the dataset (e.g., unigrams or bigrams or both).
        k : int, optional
            Number of top features to return per class (default is 5).

        Returns:
        --------
        top_features : Dataframe
            a dataframe containing the top k feature names and their coefficients for each class with counts in each class.
        """
        # Get model coefficients and feature names
        coef = self.model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        # Top k features pushing towards class 1 (most positive coefficients)
        top_k_class_1_indices = np.argsort(coef)[-k:]
        top_k_class_1_features = [feature_names[i] for i in top_k_class_1_indices[::-1]]
        top_k_class_1_coefs = [coef[i] for i in top_k_class_1_indices[::-1]]

        # Top k features pushing towards class 0 (most negative coefficients)
        top_k_class_0_indices = np.argsort(coef)[:k]
        top_k_class_0_features = [feature_names[i] for i in top_k_class_0_indices]
        top_k_class_0_coefs = [coef[i] for i in top_k_class_0_indices]

        # count all the top5 features in the reviews text to see if it makes sense
        words_to_count = top_k_class_1_features + top_k_class_0_features
        cofs = top_k_class_1_coefs + top_k_class_0_coefs

        combined_counts = utils.get_count_of_words(dataset_name=dataset_name, words=words_to_count)
        combined_counts['model_name'] = model_name
        combined_counts['dataset_name'] = dataset_name
        combined_counts['coefs'] = cofs
        combined_counts['class'] = [1] * k + [0] * k
        combined_counts['feature'] = combined_counts.index

        combined_counts = combined_counts[
            ['model_name', 'dataset_name', 'class', 'feature', 'deceptive_count', 'truthful_count', 'coefs']]
        return combined_counts


def run_the_model(dataset_name, X_train, y_train, X_test, y_test, vectorizer):
    """
    Runs the Logistic Regression model on the provided data and evaluates it.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., unigrams or bigrams or both).
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    y_train : array-like of shape (n_samples,)
        The target values.
    X_test : array-like of shape (n_samples, n_features)
        The test input samples.
    y_test : array-like of shape (n_samples,)
        The target values of the test set.
    vectorizer: CountVectorizer or TfidfVectorizer
        The vectorizer used to transform the text data.

    Returns:
    --------
    df_evaluation : pd.DataFrame
        A DataFrame with evaluation scores.
    y_pred : array-like
        Predicted class labels for the test set.
    """
    # Prepare a list to collect evaluation results
    evaluations = []
    best_y_pred = None
    max_accuracy = -1

    # loop over Regularization type ('l1' or 'l2')
    for penalty in ['l1', 'l2']:
        print(f"Running Logistic Regression with {penalty} regularization.")
        model = Logreg(penalty=penalty)
        params = model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model performance
        df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)

        model_name = f'Logistic Regression with {penalty} Penalty'

        # Get top and bottom features for analysis
        df_combined_counts = model.get_top_k_features(vectorizer, model_name, dataset_name, k=5)

        write_mode = 'w' if dataset_name == 'unigrams' else 'a'
        write_header = False if dataset_name == 'both' else True
        df_combined_counts.to_csv("../Output/top_k_features.csv", mode=write_mode, header=write_header)

        if df_scores['accuracy'] > max_accuracy:
            max_accuracy = df_scores['accuracy']
            best_y_pred = y_pred

        new_row = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            **df_scores,
            'params': str(params)
        }
        # Append new_row to evaluations list
        evaluations.append(new_row)

        # Create DataFrame from the list of evaluations
    df_evaluations = pd.DataFrame(evaluations)
    return df_evaluations, best_y_pred
