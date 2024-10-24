"""
File: utils.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - utilises functions for the entire project
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score




def create_folds(X, y, n_folds=8):
    """
    Splits the data into n stratified folds for cross-validation.
    :param X: Feature matrix.
    :param y: Labels.
    :param n_folds: Number of folds to create.
    :return: Generator yielding train and test indices for each fold.
    """
    skf = StratifiedKFold(n_splits=n_folds)
    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def calculate_scores(y_true, y_pred):
    """
    Calculates accuracy, precision, recall, and F1 scores.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Dictionary of calculated scores.
    """
    scores = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='deceptive'),
        'recall': recall_score(y_true, y_pred, pos_label='deceptive'),
        'f1': f1_score(y_true, y_pred, pos_label='deceptive')
    }
    return scores


def display_scores(scores):
    """
    Displays the performance scores.
    :param scores: Dictionary containing performance metrics.
    """
    for metric, score in scores.items():
        print(f"{metric.capitalize()}: {score:.4f}")


def cross_validate_model(model, X, y, n_folds=8):
    """
    Perform cross-validation for a given model.
    :param model: Initialized model (e.g., MultinomialNB, LogisticRegression).
    :param X: Feature matrix.
    :param y: Labels.
    :param n_folds: Number of cross-validation folds.
    :return: Average performance scores across all folds.
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
    :param df: Preprocessed DataFrame containing the cleaned text data.
    :param use_bigrams: Boolean flag to switch between unigrams and bigrams.
    :return: Feature matrix X and target vector y.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 2) if use_bigrams else (1, 1), stop_words='english')
    X = vectorizer.fit_transform(df['review'])  # Use the preprocessed text column
    y = df['is_fake']  # The target labels (truthful or deceptive)

    return X, y

def split_into_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def get_top_features(self, n=5):
    if hasattr(self.model, 'feature_importances_'):
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[-n:]
        print("Top features for the random forest:")
        print(feature_names[top_indices])
