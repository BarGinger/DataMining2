"""
File: utils.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Utilizes functions for the entire project
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


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


def extract_features(df, use_bigrams=False):
    """
    Converts the preprocessed text data into a document-term matrix with unigrams or bigrams.

    Parameters:
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing the cleaned text data.
    use_bigrams : bool
        Boolean flag to indicate whether to use bigrams (default is False).

    Returns:
    -------
    tuple
        A tuple containing:
        - Feature matrix X
        - Target vector y
        - Vectorizer for future analysis.
    """
    # we use stop_words = None because reviews were already preprocessed
    vectorizer = CountVectorizer(ngram_range=(2, 2) if use_bigrams else (1, 1), stop_words=None)
    X = vectorizer.fit_transform(df['review'])  # Use the preprocessed text column
    # Convert 'is_fake' from FAKE Enum to integers
    y = df['is_fake'].apply(lambda x: 1 if x.name == "DECEPTIVE" else 0)

    return X, y, vectorizer


def split_into_train_test(X, y):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.

    Returns:
    -------
    tuple
        A tuple containing:
        - X_train
        - X_test
        - y_train
        - y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def get_feature_range(total_features):
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
    mi_scores = mutual_info_classif(X, y, discrete_features=True)

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