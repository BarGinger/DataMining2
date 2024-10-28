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
from mlxtend.evaluate import mcnemar, mcnemar_table

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


def preprocess_text_batch(texts: list, hotel_names: list):
    """
    Preprocess a batch of texts using Named Entity Recognition (NER) and Part-of-Speech (POS) tagging
    to remove proper nouns, named entities, and hotel names.

    Parameters:
    ----------
    texts : list
        A list of text strings to preprocess.
    hotel_names : list
        A list of hotel names to be removed from the reviews.

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
    stop_words.update(spacy.lang.en.stop_words.STOP_WORDS)
    lemmatizer = WordNetLemmatizer()

    for doc, hotel_name in zip(docs, hotel_names):
        # Remove tokens identified as proper nouns (PROPN) and named entities (PERSON, GPE)
        cleaned_tokens = [
            token.text for token in doc
            if not (token.pos_ in ["PROPN"] or token.ent_type_ in ["PERSON", "GPE"])
        ]

        # Remove the hotel name if it exists in the tokens
        cleaned_tokens = [token for token in cleaned_tokens if token.lower() != hotel_name.lower()]
        # Stopword removal before lemmatization
        tokens_without_stopwords = [token for token in cleaned_tokens if token not in stop_words]

        # Lemmatization after stopword removal
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_stopwords]

        # Remove tokens that may contain numbers or undesired patterns
        lemmatized_tokens = [
            token for token in lemmatized_tokens
            if not re.fullmatch(r'\d+', token) and  # Remove tokens that are purely digits
               not re.search(r'\bth\b', token) and  # Remove tokens like 'th'
               not re.search(r'\d', token) and  # Remove tokens containing digits
               len(token) > 1  # Remove single character tokens
        ]

        # Join the tokens back into a string
        preprocessed_text = ' '.join(lemmatized_tokens)
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


def  run_all_models():
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
        # {
        #     "model_name": "Multinomial Naive Bayes",
        #     "model_run_method": MNB.run_the_model
        # },
        {
            "model_name": "Logistic Regression with Lasso Penalty",
            "model_run_method": LR.run_the_model
        },
        # {
            # "model_name": "Classification Trees",
            # "model_run_method": CLT.run_the_model
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
    
    # Dictionary to store predictions from each model
    all_model_preds = {}

    for model_info in models:
            print("######################################################")
            print(f"Running model - {model_info['model_name']}")
            model_run_method = model_info["model_run_method"]

            for dataset_name, dataset in datasets_dict.items():
                print(f"Running model - {model_info['model_name']} with {dataset_name}")
                df_evaluation, y_pred = model_run_method(dataset_name, *dataset)

                # Store predictions for statistical testing
                all_model_preds[f"{model_info['model_name']} ({dataset_name})"] = y_pred

                if df_evaluations.empty:
                    df_evaluations = df_evaluation
                else:
                    df_evaluations = pd.concat([df_evaluations, df_evaluation], ignore_index=True)

    df_evaluations.to_csv(EVALUATIONS_FILENAME, index=False)
    print(f"df_evaluations was saved into {EVALUATIONS_FILENAME}")
    print("Done with all models!")
    
    
    
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
def compare_all_models(true_labels, model_preds):
    """
    Compare four models pairwise using McNemar's test.

    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels.
    model_preds : dict
        A dictionary with model names as keys and their respective predictions as values.

    Returns:
    --------
    results : dict
        A dictionary with chi-squared statistics and p-values for all pairwise comparisons.
    """
    model_names = list(model_preds.keys())
    results = {}
    alpha = 0.05  # significance level

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            print(f"\nComparison: {model1} vs {model2}")

            cm = get_contingency_matrix(true_labels, model_preds[model1], model_preds[model2])
            chi2_stat, p_val = mcnemar_test(cm)

            # Interpret the result
            if p_val < alpha:
                print(f"p-value = {p_val:.4f}, which is less than {alpha}.")
                print(f"We reject the null hypothesis (H₀). {model1} and {model2} have significantly different accuracies.")
            else:
                print(f"p-value = {p_val:.4f}, which is greater than {alpha}.")
                print(f"We fail to reject the null hypothesis (H₀). {model1} and {model2} do not have significantly different accuracies.")
            
            results[f"{model1}_vs_{model2}"] = {'chi2': chi2_stat, 'p_value': p_val}

    return results



if __name__ == "__main__":
    # Uncomment to preprocess data before running models, we only need to run this once
    # preprocess()
    run_all_models()
    
    
    "TODO: ----------------------------------------------"
    true_labels = " "
    
    # Perform statistical testing on the predictions
    # true_labels = datasets_dict['unigrams'][3]  # Assuming y_test is the fourth item in the tuple
    # This varible will only work when we run the codes and we have the y_pred ready for each model
    compare_all_models(true_labels, )

