"""
File: main.py
Student 1: Bar Melinarskiy, student number: 2482975
Student 2: Prathik Alex Matthai, student number: 9020675
Student 3: Mohammed Bashabeeb, student number: 7060424
Date: September 12, 2024
Description: Assignment 2 - Classification for the Detection of Opinion Spam
"""

import spacy
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

import utils
import Multinomial_Naive_Bayes as MNB
import Logistic_Regression as LR
import Classification_Trees as CLT
import Random_Forests as RF

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Check if the Spacy model is already installed, download if not
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    from spacy.cli import download
    download(model_name)
    nlp = spacy.load(model_name)

PREPROCESSED_FILENAME = r"..\Data\preprocessed_df.csv"
EVALUATIONS_FILENAME = r"..\Output\df_evaluations.csv"


class FAKE(Enum):
    """ Enumeration for labeling review authenticity. """
    TRUTHFUL = 0
    DECEPTIVE = 1


def clean_text(text):
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
    text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)       # Remove numbers
    text = " ".join([word for word in text.split() if word.isalpha()])  # Remove non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    return text.lower()


def preprocess_text_batch(texts: list):
    """
    Preprocess a batch of texts using Named Entity Recognition (NER) and Part-of-Speech (POS) tagging
    to remove proper nouns and named entities.

    Parameters:
    ----------
    texts : list
        A list of text strings to preprocess.

    Returns:
    -------
    list
        A list of preprocessed text strings.
    """
    processed_texts = []
    cleaned_texts = [clean_text(text) for text in texts]  # Clean texts first to reduce workload for Spacy

    # Use Spacy's pipe for batch processing with optimized settings
    with nlp.disable_pipes("ner", "parser"):  # Disable unnecessary components for efficiency
        docs = nlp.pipe(cleaned_texts, batch_size=100, n_process=2)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for doc in docs:
        # Remove tokens identified as proper nouns (PROPN)
        cleaned_tokens = [
            token.text for token in doc if not (token.pos_ in ["PROPN"])
        ]

        # Tokenization and stopword removal
        tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens if token not in stop_words]

        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)
        processed_texts.append(preprocessed_text)

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

    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.txt'):  # Check if file is a text file
                with zip_file.open(file_info.filename) as txt_file:
                    # Determine if review is deceptive or truthful based on filename
                    if "negative" in file_info.filename:
                        text_stream = TextIOWrapper(txt_file, encoding='utf-8')
                        text = text_stream.read()
                        texts.append(text)  # Collect all texts for batch processing
                        is_fake = FAKE.DECEPTIVE if "deceptive" in file_info.filename else FAKE.TRUTHFUL
                        labels.append(is_fake)
                        filenames.append(file_info.filename)

    # Batch preprocess all the collected texts
    preprocessed_texts = preprocess_text_batch(texts)

    print(f"length of filenames: {len(filenames)}")
    print(f"length of labels: {len(labels)}")
    print(f"length of texts: {len(texts)}")
    print(f"length of preprocessed_texts: {len(preprocessed_texts)}")

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
    df = read_zip(r"..\Data\op_spam_v1.4.zip")
    df.to_csv(PREPROCESSED_FILENAME, index=False)
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


def get_datasets(use_bigrams=False):
    """
    Extract features from the preprocessed DataFrame and split it into training and testing datasets.

    Parameters:
    ----------
    use_bigrams : bool
        Whether to include bigrams in feature extraction.

    Returns:
    -------
    tuple
        A tuple containing the training and testing datasets (X_train, y_train, X_test, y_test, vectorizer).
    """
    df = get_df()
    X, y, vectorizer = utils.extract_features(df=df, use_bigrams=use_bigrams)
    X_train, X_test, y_train, y_test = utils.split_into_train_test(X=X, y=y)
    return X_train, y_train, X_test, y_test, vectorizer


def get_datasets_wrapper(use_bigrams):
    """
    Wrapper function to get datasets and store them in a dictionary.

    Parameters:
    ----------
    use_bigrams : bool
        Whether to use bigrams or not.

    Returns:
    -------
    datasets : tuple
        A tuple containing the datasets (X_train, X_test, y_train, y_test, vectorizer).
    """
    return get_datasets(use_bigrams=use_bigrams)


def run_all_models():
    """
    Run all specified models on both unigram and bigram datasets, evaluate their performance,
    and save the evaluation results to a CSV file.
    """
    # Dictionary to store datasets
    datasets_dict = {
        'unigrams': get_datasets_wrapper(use_bigrams=False),
        'bigrams': get_datasets_wrapper(use_bigrams=True)
    }

    # iterate over all the models, for each one train and predict on unigrams and bigrams datasets
    models = [
        {
            "model_name": "Multinomial Naive Bayes",
            "model_run_method": MNB.run_the_model
        },
        # {
        #     "model_name": "Logistic Regression with Lasso Penalty",
        #     "model_run_method": LR.run_the_model
        # },
        # {
        #     "model_name": "Classification Trees",
        #     "model_run_method": CLT.run_the_model
        # },
        # {
        #     "model_name": "Random Forests",
        #     "model_run_method": RF.run_the_model
        # }
    ]

    df_evaluations = pd.DataFrame(columns=[
        'model_name', 'dataset_name', 'accuracy', 'precision', 'recall', 'f1',
        'top_5_features_deceptive', 'top_5_features_truthful',
        'bottom_5_features_deceptive', 'bottom_5_features_truthful'
    ])

    for model_info in models:
        print("######################################################")
        print(f"Running model - {model_info['model_name']}")
        for dataset_name, dataset in datasets_dict.items():
            print(f"Running model - {model_info['model_name']} with {dataset_name}")
            model_run_method = model_info["model_run_method"]
            df_evaluation = model_run_method(dataset_name, *dataset)
            print(f"df_evaluation = {df_evaluation}")
            df_evaluations = pd.concat([df_evaluations, df_evaluation], ignore_index=True)

    df_evaluations.to_csv(EVALUATIONS_FILENAME, index=False)


if __name__ == "__main__":
    # Uncomment to preprocess data before running models, we only need to run this once
    # preprocess()
    run_all_models()
    print("Done with all models!")
