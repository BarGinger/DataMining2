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
        Returns the top k most influential features.
        
        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            Number of top features to return (default is 5).
        
        Returns:
        --------
        top_k_features : array
            Array containing the top k features.
        """
        feature_importance = np.abs(self.model.coef_[0])
        top_k_indices = np.argsort(feature_importance)[-k:]
        feature_names = vectorizer.get_feature_names_out()

        print(f"Top {k} features for classification: {feature_names[top_k_indices]}")
        return feature_names[top_k_indices]

    def get_bottom_k_features(self, vectorizer, k=5):
        """
        Returns the least k influential features.
        
        Parameters:
        -----------
        vectorizer : CountVectorizer or TfidfVectorizer
            The vectorizer used to transform the text data.
        k : int, optional
            Number of bottom features to return (default is 5).
        
        Returns:
        --------
        bottom_k_features : array
            Array containing the bottom k features.
        """
        feature_importance = np.abs(self.model.coef_[0])
        bottom_k_indices = np.argsort(feature_importance)[:k]
        feature_names = vectorizer.get_feature_names_out()

        print(f"Bottom {k} features for classification: {feature_names[bottom_k_indices]}")
        return feature_names[bottom_k_indices]


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

    # Get top and bottom features for analysis
    df_top_5 = model.get_top_k_features(vectorizer, k=5)
    df_bottom_5 = model.get_bottom_k_features(vectorizer, k=5)

    new_row = {
            'model_name': f'Logistic Regression with Lasso Penalty',
            'dataset_name': dataset_name,
            **df_scores,
            'top_5_features_deceptive': ", ".join(df_top_5['deceptive']),
            'top_5_features_truthful': ", ".join(df_top_5['truthful']),
            'bottom_5_features_deceptive': ", ".join(df_bottom_5['deceptive']),
            'bottom_5_features_truthful': ", ".join(df_bottom_5['truthful'])
        }

    df_evaluation = pd.DataFrame([new_row])
    return df_evaluation, y_pred
