"""
File: utils.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Utilizes functions for the entire project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from enum import Enum

PREPROCESSED_FILENAME = r"..\Data\preprocessed_df.csv"
EVALUATIONS_FILENAME = r"..\Output\df_evaluations.csv"
STATISTICAL_ANALYSIS_FILENAME = r"..\Output\df_statistical_analysis.csv"

class FAKE(Enum):
    """ Enumeration for labeling review authenticity. """
    TRUTHFUL = 0
    DECEPTIVE = 1

def get_df():
    """
    Load the preprocessed DataFrame from CSV.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the preprocessed reviews and their labels.
    """
    df = pd.read_csv(PREPROCESSED_FILENAME)
    df['is_fake'] = df['is_fake'].apply(lambda x: FAKE[x.split('.')[1]])
    return df


def create_folds(X, y, n_folds=8):
    """
    Splits the data into stratified folds for cross-validation.

    Parameters:
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels corresponding to the feature matrix.
    n_folds : int
        Number of folds to create (default is 8).

    Yields:
    ------
    tuple
        A tuple of train and test indices for each fold.
    """
    skf = StratifiedKFold(n_splits=n_folds)
    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def calculate_scores(y_true, y_pred):
    """
    Calculates various performance metrics: accuracy, precision, recall, and F1 score.

    Parameters:
    ----------
    y_true : np.ndarray
        True labels for the test set.
    y_pred : np.ndarray
        Predicted labels from the model.

    Returns:
    -------
    dict
        A dictionary containing calculated accuracy, precision, recall, and F1 scores.
    """
    scores = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=1),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return scores


def display_scores(scores):
    """
    Displays the performance scores.

    Parameters:
    ----------
    scores : dict
        Dictionary containing performance metrics.

    Returns:
    -------
    None
    """
    for metric, score in scores.items():
        print(f"{metric.capitalize()}: {score:.4f}")


def cross_validate_model(model, X, y, n_folds=8):
    """
    Performs cross-validation for a given model and returns average performance scores.

    Parameters:
    ----------
    model : object
        An initialized model (e.g., MultinomialNB, LogisticRegression).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    n_folds : int
        Number of cross-validation folds (default is 8).

    Returns:
    -------
    dict
        Average performance scores across all folds including accuracy, precision, recall, and F1 score.
    """
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for train_idx, test_idx in create_folds(X, y, n_folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_scores = calculate_scores(y_test, y_pred)

        # Append scores from the fold
        for metric, score in fold_scores.items():
            scores[metric].append(score)

    # Compute average scores over all folds
    avg_scores = {metric: np.mean(score_list) for metric, score_list in scores.items()}

    return avg_scores


def extract_features(reviews, ngram_range=(1, 1)):
    """
    Converts the preprocessed text data into a document-term matrix with unigrams or bigrams or a combo of both.

    Parameters:
    ----------
    reviews : np.ndarray
        reviews' cleaned text data.
    ngram_range : range
        range of grams to create default is unigrams (1, 1)

    Returns:
    -------
    tuple
        A tuple containing:
        - Feature matrix X
        - Vectorizer for future analysis.
    """
    # we use stop_words = None because reviews were already preprocessed
    # vectorizer = CountVectorizer(ngram_range=(2, 2) if use_bigrams else (1, 1), stop_words=None, min_df=5)
    # X = vectorizer.fit_transform(df['review'])  # Use the preprocessed text column

    vectorizer = TfidfVectorizer(min_df=0.002, max_df=0.998, ngram_range=ngram_range)
    X = vectorizer.fit_transform(reviews).toarray()

    vect = TfidfVectorizer(ngram_range=ngram_range)
    X1 = vect.fit_transform(reviews).toarray()

    no_filter_count = vect.get_feature_names_out().size
    filter_count = vectorizer.get_feature_names_out().size
    diff = no_filter_count - filter_count
    print(f"the filter removed {diff} features, from {no_filter_count} to {filter_count}")

    return X, vectorizer


def get_count_of_words(dataset_name, words):
    # Define your top features (both unigrams and bigrams) for each class
    df_reviews = get_df()
    # Initialize CountVectorizer with target features as vocabulary

    if 'both' in dataset_name:
        ngram_range = (1,2)
    elif 'unigrams' in dataset_name:
        ngram_range = (1, 1)
    else:
        ngram_range = (2, 2)

    vectorizer =  CountVectorizer(vocabulary=set(words), ngram_range=ngram_range)
    X = vectorizer.fit_transform(df_reviews['review'])  # Fit and transform on the reviews
    # Get the counts for each feature
    feature_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Separate counts by class (deceptive or truthful)
    deceptive_counts = feature_counts[df_reviews['is_fake'] == FAKE.DECEPTIVE].sum(axis=0)
    truthful_counts = feature_counts[df_reviews['is_fake'] == FAKE.TRUTHFUL].sum(axis=0)

    # Combine into a single DataFrame for comparison
    combined_counts = pd.DataFrame({
        'deceptive_count': deceptive_counts,
        'truthful_count': truthful_counts
    }).fillna(0).astype(int)

    # Reindex the DataFrame to match the order of the words list
    combined_counts = combined_counts.reindex(index=words)

    print("Word counts for top features:")
    print(combined_counts)
    return combined_counts


def get_feature_range(total_features):
    """
    Get the feature range for top k features in MNB

    Parameters:
    -----------
    total_features : int
        total number of features

    Returns:
    --------
    the range of number of features to check
    """

    initial_round_numbers = [10, 15, 25, 50, 100, 250, 500, 1000]
    step_size = 1000

    # Start with the initial round numbers
    feature_range = initial_round_numbers.copy()

    # Add values in steps of 500 until reaching the total number of features
    current_value = 1000
    while current_value + step_size <= total_features and current_value < 5000:
        current_value += step_size
        feature_range.append(current_value)

    # Ensure the last value is the total number of features
    if feature_range[-1] != total_features:
        feature_range.append(total_features)

    # Filter out any values that are higher than the total number of features
    feature_range = [value for value in feature_range if value <= total_features]

    return feature_range

def get_top_features(X, y, k=100):
    """
       Select the top k features based on mutual information with respect to the target variable.

       Parameters:
       -----------
       X : ndarray
           Feature matrix.
       y : ndarray
           Target vector.
       k : int
           Number of top features to select.

       Returns:
       --------
       X_new : ndarray
           Feature matrix with only the top k features.
       selected_indices : ndarray
           Indices of the selected features.
   """

    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, discrete_features=False)

    # Get the indices of the top k features
    top_k_indices = np.argsort(mi_scores)[-k:]

    # Sort indices in ascending order for easier feature extraction
    top_k_indices.sort()

    # Select the top k features
    X_new = X[:, top_k_indices]

    return X_new, top_k_indices


def feature_selection(X, y, method='l1', max_features=100):
    """
    Performs feature selection using L1 or L2 regularization on Logistic Regression.

    Parameters:
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    method : str
        Regularization method, 'l1' for Lasso or 'l2' for Ridge (default is 'l1').
    max_features : int
        Maximum number of features to select (default is 100).

    Returns:
    -------
    tuple
        A tuple containing:
        - Reduced feature matrix X_new
        - Indices of the selected features.
    """
    penalty = 'l1' if method == 'l1' else 'l2'
    logistic = LogisticRegression(penalty=penalty, solver='liblinear', max_iter=1000)

    # Perform cross-validation to find the optimal number of features
    logistic.fit(X, y)

    importance = np.abs(logistic.coef_).flatten()

    # Sort by importance and keep the top max_features
    selected_features_indices = np.argsort(importance)[-max_features:]
    X_new = X[:, selected_features_indices]

    return X_new, selected_features_indices