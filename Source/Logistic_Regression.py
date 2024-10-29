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
           None
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
        Returns the top k most influential features.
        
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
        model.train(X_train, y_train)
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
            'top_5_features': ", ".join(df_top_5),
            'bottom_5_features': ", ".join(df_bottom_5)
        }

        # Create DataFrame from the list of evaluations
    df_evaluations = pd.DataFrame(evaluations)
    return df_evaluations, best_y_pred
