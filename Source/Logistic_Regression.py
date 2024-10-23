import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from main import get_df


class Logreg:
    def __init__(self):
        self.vectorizer = None
        # Logistic Regression with L1 penalty (Lasso)
        self.model = Logreg(penalty='l1', solver='liblinear', random_state=42)

    # Feature Extraction: Unigram and Bigram features
    def extraction(self, X_train, X_test, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec

    # Evaluation Function
    def evaluate(self, X_train, X_test, y_train, y_test):
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
        if hasattr(self.model, 'coef_'):
            feature_names = np.array(self.vectorizer.get_feature_names_out())
            coefficients = self.model.coef_.flatten()
            top_indices = np.argsort(np.abs(coefficients))[-n:]
            print("Top features for Logistic Regression with Lasso:")
            for index in top_indices:
                print(f"{feature_names[index]}: {coefficients[index]}")

    def run(self, df_positive, df_negative):
        # Load dataset
        df = pd.concat([df_positive, df_negative])
        X = df['review']
        
        # Convert 'is_fake' from FAKE Enum to integers
        y = df['is_fake'].apply(lambda x: 1 if x.name == "DECEPTIVE" else 0)

        # Split the data: 80% for training (Folds 1-4), 20% for testing (Fold 5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Run for Unigrams
        print("=== Results for Unigrams ===")
        X_train_uni, X_test_uni = self.extract_features(X_train, X_test, ngram_range=(1, 1))
        self.evaluate_model(X_train_uni, X_test_uni, y_train, y_test)

        # Get top features 
        print("\nTop features for Logistic Regression with Lasso (Unigrams):")
        self.get_top_features()

        # Run for Bigrams
        print("\n=== Results for Bigrams ===")
        X_train_bi, X_test_bi = self.extract_features(X_train, X_test, ngram_range=(1, 2))
        self.evaluate_model(X_train_bi, X_test_bi, y_train, y_test)

        # Get top features for bigrams
        print("\nTop features for Logistic Regression with Lasso (Bigrams):")
        self.get_top_features()



if __name__ == "__main__":
    # Load the dataset from the get_df function
    df_positive, df_negative = get_df()

    # Initialize and run the DecisionTreeModel
    log_model = Logreg()
    log_model.run(df_positive, df_negative)

    print("Done!")