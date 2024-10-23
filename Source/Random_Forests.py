import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from main import get_df


class RandomForestModel:
    def __init__(self):
        self.vectorizer = None
        self.model = RandomForestClassifier(random_state=42)

    # Feature Extraction: Unigram and Bigram features
    def extract_features(self, X_train, X_test, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    # Evaluation Function using Cross-Validation
    def evaluate_model_cv(self, X_train, y_train, cv=5):
        scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Cross-Validation Accuracy Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
        return scores

    # Final Evaluation on Test Set
    def evaluate_model_test(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return accuracy, precision, recall, f1

    # Get top features
    def get_top_features(self, n=5):
        if hasattr(self.model, 'feature_importances_'):
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-n:]
            print("Top features for the random forest:")
            print(feature_names[top_indices])

    def run(self, df):
        X = df['review']
        
        # Convert 'is_fake' from FAKE Enum to integers
        y = df['is_fake'].apply(lambda x: 1 if x.name == "DECEPTIVE" else 0)

        # Split the data: 80% for training (Folds 1-4), 20% for testing (Fold 5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Run for Unigrams
        print("=== Results for Unigrams ===")
        X_train_uni, X_test_uni = self.extract_features(X_train, X_test, ngram_range=(1, 1))
        
        # Perform Cross-Validation on Training Set
        self.evaluate_model_cv(X_train_uni, y_train, cv=5)

        # Final Evaluation on Test Set
        self.evaluate_model_test(X_train_uni, X_test_uni, y_train, y_test)

        # Get top features
        self.get_top_features()

        # Run for Bigrams
        print("\n=== Results for Bigrams ===")
        X_train_bi, X_test_bi = self.extract_features(X_train, X_test, ngram_range=(1, 2))
        
        # Perform Cross-Validation on Training Set with Bigrams
        self.evaluate_model_cv(X_train_bi, y_train, cv=5)

        # Final Evaluation on Test Set with Bigrams
        self.evaluate_model_test(X_train_bi, X_test_bi, y_train, y_test)


if __name__ == "__main__":
    # Load the dataset from the get_df function
    df = get_df()

    # Initialize and run the RandomForestModel
    rf_model = RandomForestModel()
    rf_model.run(df)

    print("Done!")
