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


class FAKE(Enum):
    TRUTHFUL = 0
    DECEPTIVE = 1


def clean_text(text):
    """ Clean text by removing HTML, numbers, and punctuation, and converting to lowercase. """
    text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)       # Remove numbers
    text = " ".join([word for word in text.split() if word.isalpha()])   # Remove non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    return text.lower()


def preprocess_text_batch(texts: list):
    """
    Preprocess a batch of texts using NER and POS tagging to remove proper nouns and named entities.
    """
    processed_texts = []

    # Clean texts first to reduce the workload for Spacy
    cleaned_texts = [clean_text(text) for text in texts]

    # Use Spacy's pipe for batch processing with optimized settings
    # Experiment with n_process and batch_size
    with nlp.disable_pipes("ner", "parser"):  # Disable unnecessary components
        docs = nlp.pipe(cleaned_texts, batch_size=100, n_process=2)  # Experiment with batch size and processes

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for doc in docs:
        # Remove tokens identified as proper nouns (PROPN)
        cleaned_tokens = [
            token.text for token in doc
            if not (token.pos_ in ["PROPN"])
        ]

        # Tokenization and stopword removal in one step
        tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens if token not in stop_words]

        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)
        processed_texts.append(preprocessed_text)

    return processed_texts



def read_zip(zip_file_path: str) -> pd.DataFrame:
    """
    Read the csv of the data from the given zip file path and preprocess the reviews.
    """
    texts = []
    labels = []
    filenames = []

    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        # Iterate over each file in the zip archive
        for file_info in zip_file.infolist():
            if file_info.filename.endswith('.txt'):  # Check if file is a text file
                with zip_file.open(file_info.filename) as txt_file:
                    # Determine if review is deceptive or truthful based on filename
                    if "negative" in file_info.filename:
                        text_stream = TextIOWrapper(txt_file, encoding='utf-8')
                        text = text_stream.read()
                        texts.append(text)  # Collect all texts for batch processing
                        if "deceptive" in file_info.filename:
                            is_fake = FAKE.DECEPTIVE
                        else:  # truthful
                            is_fake = FAKE.TRUTHFUL
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
    df = read_zip(r"..\Data\op_spam_v1.4.zip")
    df.to_csv(PREPROCESSED_FILENAME, index=False)
    return df

def get_df():
    df = pd.read_csv(PREPROCESSED_FILENAME)
    df['is_fake'] = df['is_fake'].apply(lambda x: FAKE[x.split('.')[1]])

    return df


if __name__ == "__main__":
    # preprocess()
    df = get_df()
    print("Done!")

