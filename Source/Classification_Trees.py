import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import utils



class DecisionTreeModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None

    def train(self, X_train, y_train, alphas=np.arange(0, 15, 0.5), cv=5):
        """
        Trains the Decision Tree Classifier model using cross-validation to find the best alpha.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,)
            The target values.
        alphas : list, optional
            The list of alpha values to try. Default is [0.0, 0.001, 0.01, 0.1].
        cv : int, optional
            The number of cross-validation folds. Default is 5.

        Returns:
        --------
        None
        """

        # Define the hyperparameter grid
        param_grid = {
            'ccp_alpha': alphas,
            'random_state': [42],
            'max_depth': np.arange(10, 80, 10),
            'min_samples_split': np.arange(10, 50, 5),
            'min_samples_leaf': np.arange(5, 100, 5)
        }

        # Perform grid search with cross-validation
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Save the best model after cross-validation
        self.model = grid.best_estimator_
        print(f"Best ccp_alpha: {self.model.ccp_alpha}")
        print(f"Best max_depth: {self.model.max_depth}")
        print(f"Best min_samples_split: {self.model.min_samples_split}")
        print(f"Best min_samples_leaf: {self.model.min_samples_leaf}")

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

        # Get top features
    def get_top_features(self, vectorizer, k=5):
        """
        Returns the top k features based on feature importances from the Decision Tree model.

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
        if hasattr(self.model, 'feature_importances_'):
            feature_names = np.array(vectorizer.get_feature_names_out())
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-k:]

            # Extract feature names
            # feature_names = np.array(vectorizer.get_feature_names_out())
            # # Get feature importances from the model
            # importances = self.model.feature_importances_
            #
            # # Get the top k most important features
            # top_k_indices = np.argsort(importances)[-k:]  # Top k indices
            # top_k_features = feature_names[top_k_indices]

            return {
                'deceptive': feature_names[top_indices],  # Example key: might need specific class labels
                'truthful': []#top_k_features
            }
        else:
            raise Exception("Model does not have feature importances. Make sure the model is trained.")

    def get_bottom_k_features(self, vectorizer, k=5):
        """
        Returns the bottom k features based on feature importances from the Decision Tree model.

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
        if hasattr(self.model, 'feature_importances_'):
            # Extract feature names
            feature_names = np.array(vectorizer.get_feature_names_out())
            # Get feature importances from the model
            importances = self.model.feature_importances_

            # Get the bottom k least important features
            bottom_k_indices = np.argsort(importances)[:k]  # Bottom k indices
            bottom_k_features = feature_names[bottom_k_indices]

            return {
                'deceptive': bottom_k_features,  # Example key: might need specific class labels
                'truthful': bottom_k_features
            }
        else:
            raise Exception("Model does not have feature importances. Make sure the model is trained.")


    # Feature Extraction: Unigram and Bigram features
    # def extract_features(self, X_train, X_test, ngram_range=(1, 1)):
    #     self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=5000)
    #     X_train_vec = self.vectorizer.fit_transform(X_train)
    #     X_test_vec = self.vectorizer.transform(X_test)
    #     return X_train_vec, X_test_vec
    #
    # Evaluation Function
    # def evaluate_model(self, X_train, X_test, y_train, y_test):
    #     self.model.fit(X_train, y_train)
    #     y_pred = self.model.predict(X_test)
    #
    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #
    #     print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    #     return accuracy, precision, recall, f1
    #
    # def run(self, df):
    #     X = df['review']
    #
    #     # Convert 'is_fake' from FAKE Enum to integers
    #     y = df['is_fake'].apply(lambda x: 1 if x.name == "DECEPTIVE" else 0)
    #
    #     # Split the data: 80% for training (Folds 1-4), 20% for testing (Fold 5)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #
    #     # Run for Unigrams
    #     print("=== Results for Unigrams ===")
    #     X_train_uni, X_test_uni = self.extract_features(X_train, X_test, ngram_range=(1, 1))
    #     self.evaluate_model(X_train_uni, X_test_uni, y_train, y_test)
    #
    #     # Get top features
    #     print("\nTop features for the Classification Tree:")
    #     self.get_top_features()
    #
    #     # Run for Bigrams
    #     print("\n=== Results for Bigrams ===")
    #     X_train_bi, X_test_bi = self.extract_features(X_train, X_test, ngram_range=(1, 2))
    #     self.evaluate_model(X_train_bi, X_test_bi, y_train, y_test)


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
    # todo adopt this
    model = DecisionTreeModel()
    model.train(X_train, y_train)
    # predict on given test set
    y_pred = model.predict(X_test)
    # Calculate scores
    df_scores = utils.calculate_scores(y_true=y_test, y_pred=y_pred)
    # Get top and bottom 5 features
    df_top_5 = model.get_top_features(vectorizer, k=5)
    df_bottom_5 = model.get_bottom_k_features(vectorizer, k=5)

    # Create DataFrame from the list of evaluations
    df_evaluations = pd.DataFrame([{
        'model_name': f'Decision Tree',
        'dataset_name': dataset_name,
        **df_scores,  # Unpack scores
        'top_5_features_deceptive': ", ".join(df_top_5['deceptive']),
        'top_5_features_truthful': ", ".join(df_top_5['truthful']),
        'bottom_5_features_deceptive': ", ".join(df_bottom_5['deceptive']),
        'bottom_5_features_truthful': ", ".join(df_bottom_5['truthful'])
    }])
    return df_evaluations
