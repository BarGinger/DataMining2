import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import utils


class DecisionTreeModel:
    """
    A class to represent DecisionTreeModel for text classification.

    Attributes:
    -----------
    model : DecisionTreeClassifier
        The DecisionTreeModel instance.
    max_features : int
        The maximum number of features to be selected. Default is 100.
    selected_features_indices : list
        The indices of the selected features.

    Methods:
    --------
    train(X_train, y_train, alphas=[0.1, 0.5, 1.0], cv=5)
        Trains the model using cross-validation to find the best parameters.
    predict(X)
        Predicts the class labels for the input data.
    get_top_k_features(vectorizer, k=5)
        Returns the top k features for each class.
    get_bottom_k_features(vectorizer, k=5)
        Returns the least k features for each class.
    """

    def __init__(self):
        self.model = None


    def train(self, X_train, y_train, max_depth=[3, 5, 7, 10], cv=5, max_features=[10, 100, 500]):
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
        k : int, optional
            The number of features to use. Default is 100.
        max_features: int
            The max_features for the model training
        Returns:
        --------
        None
        """
        # Train the Multinomial Naive Bayes model using cross-validation to find the best alpha
        param_grid = {'max_depth': max_depth, 'max_features': max_features}  # Cross-validate over different alpha values
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv, scoring='accuracy')
        grid.fit(X_train, y_train)

        # Save the best model after cross-validation
        self.model = grid.best_estimator_
        print(f"Best max_depth: {self.model.max_depth}")
        print(f"Best max_features: {self.model.max_features}")

        # Save cv_results_ to a CSV file
        logs = pd.DataFrame(grid.cv_results_)

        return {
            'cv': cv,
            'scoring': 'accuracy',
            'max_depth': self.model.max_depth,
            'max_features': self.model.max_features
        }, logs
        
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

        y_pred = self.model.predict(X)
        return y_pred

def run_the_model(dataset_name, X_train, y_train, X_test, y_test, vectorizer):
    """
    Run the model with the given train and test sets.

    Parameters:
    -----------
    dataset_name : str
        The name of the given dataset (unigrams or bigrams or both).
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
    df_evaluation : DataFrame
        A DataFrame with evaluation scores and needed info for further analysis.
    y_pred : array-like of shape (n_samples,)
        The predicted labels for the test set.
    """
    evaluations = []
    total_count_features = X_train.shape[1]
    number_feature_range = utils.get_feature_range(total_count_features)
    best_y_pred = None
    max_accuracy = -1

    model = DecisionTreeModel()
    params, logs = model.train(X_train, y_train, max_depth=[3, 5, 7, 10], cv=5, max_features=number_feature_range)

    write_mode = 'w' if (dataset_name == 'unigrams') else 'a'
    write_header = write_mode == 'w'
    # save training logs
    logs['dataset_name'] = dataset_name
    logs.to_csv("../Output/cf_grid_search_results.csv", index=False, mode=write_mode,
                header=write_header)

    y_pred = model.predict(X_test)
    df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)

    if df_scores['accuracy'] > max_accuracy:
        max_accuracy = df_scores['accuracy']
        best_y_pred = y_pred

    new_row = {
        'model_name': f'Classification Tree',
        'dataset_name': dataset_name,
        **df_scores,
        'params': str(params)
    }
    evaluations.append(new_row)

    df_evaluations = pd.DataFrame(evaluations)
    return df_evaluations, best_y_pred  # return both evaluation dataframe and predictions of best model



