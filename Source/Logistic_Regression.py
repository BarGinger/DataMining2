from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import utils


class Logreg:
    """
    Logistic Regression model with L1 (Lasso) or L2 (Ridge) regularization for text classification.

    Attributes:
    -----------
    model : LogisticRegression
        The logistic regression model with the specified penalty.
    penalty : str
        The regularization type ('l1' for Lasso or 'l2' for Ridge).

    Methods:
    --------
    train(X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5)
        Trains the Logistic Regression model using cross-validation to find the best C.
    predict(X)
        Predicts the class labels for the input data.
    get_top_k_features(vectorizer, k=5)
        Returns the top k most influential features for each class (deceptive and truthful).
    get_bottom_k_features(vectorizer, k=5)
        Returns the bottom k least influential features for each class (deceptive and truthful).
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

    def train(self, X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5):
        """
        Trains the Logistic Regression model using cross-validation to find the best C.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data features.
        y_train : array-like of shape (n_samples,)
            Training data labels.
        Cs : list, optional
            List of C values for cross-validation (default is [0.01, 0.1, 1.0, 10]).
        cv : int, optional
            Number of cross-validation folds (default is 5).
        
        Returns:
        --------
        None
        """
        # Select the solver based on penalty type
        solver = 'liblinear' if self.penalty == 'l1' else 'lbfgs'

        # Use GridSearchCV to find the optimal C value
        param_grid = {'C': Cs}
        grid = GridSearchCV(LogisticRegression(penalty=self.penalty, solver=solver, max_iter=1000),
                            param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Store the best model after cross-validation
        self.model = grid.best_estimator_
        print(f"Best C: {self.model.C}")

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

        y_pred = self.model.predict(X)
        return y_pred

    def get_top_k_features(self, vectorizer, k=5):
        """
        Returns the top k most influential features for each class (deceptive and truthful).
        
        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            Number of top features to return (default is 5).
        
        Returns:
        --------
        dict
            A dictionary containing the top k features for each class (deceptive and truthful).
        """
        feature_importance = self.model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        # Sort features for deceptive (positive) and truthful (negative) influence
        top_k_deceptive = np.argsort(feature_importance)[-k:]  # Highest positive weights for deceptive
        top_k_truthful = np.argsort(-feature_importance)[-k:]  # Highest negative weights for truthful

        return {
            'deceptive': feature_names[top_k_deceptive],
            'truthful': feature_names[top_k_truthful]
        }

    def get_bottom_k_features(self, vectorizer, k=5):
        """
        Returns the least k influential features for each class (deceptive and truthful).
        
        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            Number of bottom features to return (default is 5).
        
        Returns:
        --------
        dict
            A dictionary containing the bottom k features for each class (deceptive and truthful).
        """
        feature_importance = self.model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        # Sort features with the least influence for each class
        bottom_k_deceptive = np.argsort(feature_importance)[:k]  # Lowest positive weights for deceptive
        bottom_k_truthful = np.argsort(-feature_importance)[:k]  # Lowest negative weights for truthful

        return {
            'deceptive': feature_names[bottom_k_deceptive],
            'truthful': feature_names[bottom_k_truthful]
        }


def run_the_model(dataset_name, X_train, y_train, X_test, y_test, vectorizer, penalty='l1'):
    """
    Runs the Logistic Regression model on the provided data and evaluates it.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., unigrams or bigrams).
    X_train : array-like
        Training data features.
    y_train : array-like
        Training data labels.
    X_test : array-like
        Test data features.
    y_test : array-like
        Test data labels.
    vectorizer : CountVectorizer or TfidfVectorizer
        Vectorizer used to create the feature matrix.
    penalty : str, optional
        Regularization type ('l1' or 'l2'). Default is 'l1'.
    
    Returns:
    --------
    df_evaluation : pd.DataFrame
        DataFrame with evaluation scores.
    y_pred : array-like
        Predicted class labels for the test set.
    """
    print(f"Running Logistic Regression with {penalty} regularization.")
    model = Logreg(penalty=penalty)
    model.train(X_train, y_train, Cs=[0.01, 0.1, 1.0, 10], cv=5)
    y_pred = model.predict(X_test)

    # Evaluate model performance
    df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)

    # Get top and bottom features for deceptive and truthful classifications
    df_top_5 = model.get_top_k_features(vectorizer, k=5)
    df_bottom_5 = model.get_bottom_k_features(vectorizer, k=5)

    # Create an evaluation row in the format used for Naive Bayes
    new_row = {
        'model_name': f'Logistic Regression ({penalty.upper()} regularization)',
        'dataset_name': dataset_name,
        **df_scores,
        'top_5_features_deceptive': ", ".join(df_top_5['deceptive']),
        'top_5_features_truthful': ", ".join(df_top_5['truthful']),
        'bottom_5_features_deceptive': ", ".join(df_bottom_5['deceptive']),
        'bottom_5_features_truthful': ", ".join(df_bottom_5['truthful'])
    }

    df_evaluation = pd.DataFrame([new_row])
    return df_evaluation, y_pred
