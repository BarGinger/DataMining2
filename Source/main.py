"""
File: main.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Classification for the Detection of Opinion Spam
"""
from operator import index

import spacy
from tabulate import tabulate
import os
import nltk
import re
import zipfile
import numpy as np
import pandas as pd
from io import TextIOWrapper
from enum import Enum
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.optimize import differential_evolution
from mlxtend.evaluate import mcnemar, mcnemar_table
from sklearn.model_selection import train_test_split

import utils
import Multinomial_Naive_Bayes as MNB
import Logistic_Regression as LR
import Classification_Trees as CLT
import Random_Forests as RF

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Check if the Spacy model is already installed, download if not
model_to_download = "en_core_web_sm"
try:
    nlp = spacy.load(model_to_download)
except OSError:
    from spacy.cli import download
    download(model_to_download)
    nlp = spacy.load(model_to_download)

PREPROCESSED_FILENAME = r"..\Data\preprocessed_df.csv"
EVALUATIONS_FILENAME = r"..\Output\df_evaluations.csv"
STATISTICAL_ANALYSIS_FILENAME = r"..\Output\df_statistical_analysis.csv"


class FAKE(Enum):
    """ Enumeration for labeling review authenticity. """
    TRUTHFUL = 0
    DECEPTIVE = 1


import re
import pandas as pd
import zipfile
from io import TextIOWrapper
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, numbers, and punctuation, and converting to lowercase.

    Parameters:
    ----------
    text : str
        The input text to be cleaned.

    Returns:
    -------
    str
        The cleaned text.
    """
    # Replace contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'d", " would", text)

    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)

    # Replace numbers with space (instead of empty string)
    text = re.sub(r'\d+', ' ', text)

    # Replace punctuation with space (instead of removing directly)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def preprocess_text_batch(texts: List[str], hotel_names: List[str]) -> List[str]:
    """
    Preprocess a batch of texts using NER and POS tagging to remove proper nouns,
    named entities, and hotel names.

    Parameters:
    ----------
    texts : List[str]
        A list of text strings to preprocess.
    hotel_names : List[str]
        A list of hotel names to be removed from the reviews.

    Returns:
    -------
    List[str]
        A list of preprocessed text strings.
    """
    processed_texts = []
    cleaned_texts = [clean_text(text) for text in texts]

    # Use Spacy's pipe for batch processing
    with nlp.disable_pipes("ner", "parser"):
        docs = nlp.pipe(cleaned_texts, batch_size=100, n_process=2)

    # Get stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(spacy.lang.en.stop_words.STOP_WORDS)

    # Remove certain words from stop words that might be important for sentiment
    words_to_keep = {'not', 'no', 'nor', 'neither', 'never', 'none'}
    stop_words = stop_words - words_to_keep

    lemmatizer = WordNetLemmatizer()

    for doc, hotel_name in zip(docs, hotel_names):
        # Process tokens with improved filtering
        tokens = []
        for token in doc:
            # Skip if token is a proper noun or named entity
            if token.pos_ in ["PROPN"] or token.ent_type_ in ["PERSON", "GPE"]:
                continue

            text = token.text.lower()

            # Skip if token matches hotel name
            if text == hotel_name.lower():
                continue

            # Skip if token is in stop words
            if text in stop_words:
                continue

            # Skip if token contains unwanted patterns
            if (len(text) <= 1 or  # Single characters
                    text.isdigit() or  # Pure digits
                    re.search(r'\bth\b', text) or  # 'th'
                    re.search(r'\d', text)):  # Contains any digits
                continue

            # Lemmatize the token
            lemmatized = lemmatizer.lemmatize(text)

            # Only add if the lemmatized form is valid
            if len(lemmatized) > 1 and lemmatized.isalpha():
                tokens.append(lemmatized)

        # Join tokens with space and add to results
        processed_text = ' '.join(tokens)
        processed_texts.append(processed_text)

    return processed_texts


def read_zip(zip_file_path: str) -> pd.DataFrame:
    """
    Read the CSV of the data from the given zip file path and preprocess the reviews.

    Parameters:
    ----------
    zip_file_path : str
        The path to the zip file containing the reviews.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the filenames, labels, and preprocessed reviews.
    """
    texts = []
    labels = []
    filenames = []
    hotel_names = []  # To store hotel names

    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.txt'):  # Check if file is a text file
                with zip_file.open(file_info.filename) as txt_file:
                    # Determine if review is deceptive or truthful based on filename
                    if "negative" in file_info.filename:
                        # Determine if review is deceptive or truthful based on filename
                        is_fake = FAKE.DECEPTIVE if "deceptive" in file_info.filename else FAKE.TRUTHFUL
                        # Extract hotel name from filename
                        hotel_name = file_info.filename.split('/')[-1].split('_')[1]  # The hotel name is the second part
                        hotel_names.append(hotel_name)

                        text_stream = TextIOWrapper(txt_file, encoding='utf-8')
                        text = text_stream.read()
                        texts.append(text)  # Collect all texts for batch processing
                        labels.append(is_fake)
                        filenames.append(file_info.filename)

    # Batch preprocess all the collected texts along with their corresponding hotel names
    preprocessed_texts = preprocess_text_batch(texts, hotel_names)

    print(f"{len(preprocessed_texts)} reviews were preprocessed")

    # Create the final DataFrame
    df = pd.DataFrame({
        "__filename__": filenames,
        "is_fake": labels,
        "review": preprocessed_texts
    })

    return df


def preprocess():
    """
    Read the data from a zip file and preprocess it, saving the results to a CSV file.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the preprocessed data.
    """
    print("Running preprocess ... this could take awhile so go and drink coffee")
    df = read_zip(r"..\Data\op_spam_v1.4.zip")
    df.to_csv(PREPROCESSED_FILENAME, index=False)
    print(f"preprocess data was saved into the following file: {PREPROCESSED_FILENAME}")
    return df


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


def get_datasets():
    """
    Extract features from the preprocessed DataFrame and split it into training and testing datasets.

    Returns:
    -------
    tuple
        A tuple containing the training and testing datasets (X_train, y_train, X_test, y_test).
    """
    df = get_df()
    X, y = df['review'], df['is_fake'].apply(lambda x: 1 if x.name == "DECEPTIVE" else 0)

    X_unigrams, vectorizer_unigram = utils.extract_features(reviews=X, ngram_range= (1, 1))
    X_bigrams, vectorizer_bigrams = utils.extract_features(reviews=X, ngram_range=(2, 2))
    X_both, vectorizer_both = utils.extract_features(reviews=X, ngram_range=(1, 2))

    _, _, train_indices, test_indices = train_test_split(
        X_unigrams, range(len(X_unigrams)), test_size=0.2, random_state=42)

    # Use the indices to split both unigram and bigram datasets
    train_unigrams = X_unigrams[train_indices]
    test_unigrams = X_unigrams[test_indices]

    train_bigrams = X_bigrams[train_indices]
    test_bigrams = X_bigrams[test_indices]

    train_both = X_both[train_indices]
    test_both = X_both[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    datasets_dict = {
        'unigrams': (train_unigrams, y_train, test_unigrams, y_test, vectorizer_unigram),
        'bigrams': (train_bigrams, y_train, test_bigrams, y_test, vectorizer_bigrams),
        'both': (train_both, y_train, test_both, y_test, vectorizer_both)
    }

    return datasets_dict


def  run_all_models():
    """
    Run all specified models on both unigram and bigram and a combo of both datasets, evaluate their performance,
    and save the evaluation results to a CSV file.

    Returns:
    --------
    models: dictionary
        A dictionary with model names as keys ({model_name}_{dataset_name}) and their respective predictions as values.
    datasets_dict: dictionary
        A dictionary containing the train and test datasets for both unigrams and bigrams and a combo of both
    """

    datasets_dict = get_datasets()

    # iterate over all the models, for each one train and predict on unigrams and bigrams datasets and a combo of both
    models = [
        {
            "model_name": "Multinomial Naive Bayes",
            "model_run_method": MNB.run_the_model
        },
        {
            "model_name": "Logistic Regression",
            "model_run_method": LR.run_the_model
        },
        {
            "model_name": "Classification Trees",
            "model_run_method": CLT.run_the_model
        },
        {
            "model_name": "Random Forests",
            "model_run_method": RF.run_the_model
        }
    ]

    df_evaluations = pd.DataFrame(columns=[
        'model_name', 'dataset_name', 'accuracy', 'precision', 'recall', 'f1'
    ])

    df_4_statistical_analysis = {}

    for model_info in models:
            print("######################################################")
            model_name = model_info['model_name']
            print(f"Running model - {model_name}")
            model_run_method = model_info["model_run_method"]

            for dataset_name, dataset in datasets_dict.items():
                print(f"Running model - {model_name} with {dataset_name}")
                df_evaluation, y_pred = model_run_method(dataset_name, *dataset)

                # Store predictions for statistical testing
                df_4_statistical_analysis[f'{model_name}_{dataset_name}'] = y_pred

                if df_evaluations.empty:
                    df_evaluations = df_evaluation
                else:
                    df_evaluations = pd.concat([df_evaluations, df_evaluation], ignore_index=True)

    df_evaluations.to_csv(EVALUATIONS_FILENAME, index=False)
    print(f"df_evaluations was saved into {EVALUATIONS_FILENAME}")
    print(tabulate(df_evaluations, headers='keys', tablefmt='pretty'))
    
    print("Done with training all models!")
    return df_4_statistical_analysis, datasets_dict
    
    
    
# Function to create the contingency matrix
def get_contingency_matrix(true_labels, model1_preds, model2_preds):
    """
    Computes the contingency matrix for two models' predictions using mlxtend's mcnemar_table.

    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels.
    model1_preds : array-like
        Predictions from the first model.
    model2_preds : array-like
        Predictions from the second model.

    Returns:
    --------
    contingency_matrix : 2x2 numpy array
        The contingency matrix [[a, b], [c, d]].
    """
    return mcnemar_table(y_target=true_labels, y_model1=model1_preds, y_model2=model2_preds)



# Function to perform McNemar's test
def mcnemar_test(contingency_matrix):
    """
    Performs McNemar's test for two models' predictions using mlxtend's mcnemar function.

    Parameters:
    -----------
    contingency_matrix : array-like
        A 2x2 numpy array [[a, b], [c, d]].

    Returns:
    --------
    chi2_stat : float
        The chi-squared statistic.
    p_value : float
        The p-value corresponding to the chi-squared statistic.
    """
    chi2_stat, p_value = mcnemar(ary=contingency_matrix, corrected=True)
    return chi2_stat, p_value



# Function to compare all four models in pairs using McNemar's test
import pandas as pd

def compare_all_models(datasets_dict, model_preds):
    """
    Compare four models pairwise using McNemar's test.

    Parameters:
    -----------
    datasets_dict : dict
        Dictionary of the different datasets used in this program.
    model_preds : dict
        A dictionary with model names as keys and their respective predictions as values.

    Returns:
    --------
    df_statistical_analysis : pd.DataFrame
        A DataFrame with chi-squared statistics and p-values for all pairwise comparisons.
    """
    model_names = list(model_preds.keys())
    results = []
    alpha = 0.05  # significance level

    # Assuming 'unigrams' dataset and index [3] for true labels
    true_labels = datasets_dict['unigrams'][3]

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            print(f"\nComparison: {model1} vs {model2}")

            # Obtain the contingency matrix and run McNemar's test
            cm = get_contingency_matrix(true_labels, model_preds[model1], model_preds[model2])
            chi2_stat, p_val = mcnemar_test(cm)

            # Interpret the result
            if p_val < alpha:
                print(f"p-value = {p_val:.4f}, which is less than {alpha}.")
                print(f"We reject the null hypothesis (H₀). {model1} and {model2} have significantly different accuracies.")
            else:
                print(f"p-value = {p_val:.4f}, which is greater than {alpha}.")
                print(f"We fail to reject the null hypothesis (H₀). {model1} and {model2} do not have significantly different accuracies.")

            # Append results as a dictionary for each pair
            results.append({
                'model1': model1,
                'model2': model2,
                'chi2': chi2_stat,
                'p_value': p_val
            })

    # Convert results to a DataFrame
    df_statistical_analysis = pd.DataFrame(results)
    df_statistical_analysis.to_csv(STATISTICAL_ANALYSIS_FILENAME, index=False)
    return df_statistical_analysis


if __name__ == "__main__":
    # Uncomment to preprocess data before running models, we only need to run this once
    # preprocess()
    df_4_statistical_analysis, datasets_dict = run_all_models()
    
    # Perform statistical testing on the predictions
    # true_labels = datasets_dict['unigrams'][3]  # Assuming y_test is the fourth item in the tuple
    # This variable will only work when we run the codes and we have the y_pred ready for each model
    compare_all_models(datasets_dict, df_4_statistical_analysis)

