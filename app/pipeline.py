import nltk
import numpy as np
import os
import pandas as pd
import re
import string
import tensorflow as tf

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import TFDistilBertForSequenceClassification

from model import model_fitting, word_embeddings, generate_test, generate_input
from preprocessing import preprocessing
from scraper import scrape_webpage
from utils import MODEL_DIR, DATA_DIR


def training_pipeline():
    nltk.download('omw-1.4')
    print("Reading training data")
    train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), names=[
                             'polarity', 'title', 'review'], header=None)
    print("Reading test data")
    test_data = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), names=[
                            'polarity', 'title', 'review'], header=None)

    # changing the values
    train_data['polarity'] = train_data['polarity'].replace(1, 0)  # negative
    train_data['polarity'] = train_data['polarity'].replace(2, 1)  # positive
    test_data['polarity'] = test_data['polarity'].replace(1, 0)  # negative
    test_data['polarity'] = test_data['polarity'].replace(2, 1)  # positive

    # Randomly choose 50000 training samples from each polarity => total with 100000 samples (having 3600000 would be have high computational cost)
    negative_reviews = train_data[train_data['polarity'] == 0].sample(
        n=50000).reset_index(drop=True)
    positive_reviews = train_data[train_data['polarity'] == 1].sample(
        n=50000).reset_index(drop=True)
    train_data = pd.concat(
        [negative_reviews, positive_reviews]).reset_index(drop=True)
    train_data = shuffle(train_data.reset_index(drop=True))
    print("Preprocessing the training data")
    train_data = preprocessing(train_data)

    # Randomly choose 20000 test samples from each polarity
    negative_reviews_test = test_data[test_data['polarity'] == 0].sample(
        n=10000).reset_index(drop=True)
    positive_reviews_test = test_data[test_data['polarity'] == 1].sample(
        n=10000).reset_index(drop=True)
    test_data = pd.concat(
        [negative_reviews_test, positive_reviews_test]).reset_index(drop=True)
    test_data = shuffle(test_data.reset_index(drop=True))
    print("Preprocessing the test data")
    test_data = preprocessing(test_data)

    # train, val, test split
    train_X = pd.DataFrame(train_data.loc[:, "text"])
    train_y = pd.DataFrame(train_data.loc[:, "polarity"])
    print("Train, Validation, and Test Split")
    X_train, X_val, y_train, y_val = train_test_split(
        train_X, train_y, test_size=0.2, random_state=100)
    X_test = pd.DataFrame(test_data.loc[:, "text"])
    y_test = pd.DataFrame(test_data.loc[:, "polarity"])

    print("Getting Word Embeddings dataframe for training dataset")
    X_train_encodings_df = word_embeddings(X_train, y_train)
    print("Done getting Word Embeddings dataframe for training dataset")

    print("Getting Word Embeddings dataframe for validation dataset")
    X_val_encodings_df = word_embeddings(X_val, y_val)
    print("Done getting Word Embeddings dataframe for validation dataset")

    print("Fitting the model...")
    #train_history = model_fitting(X_train_encodings_df, X_val_encodings_df)

    print("Generating the predictions with the test dataset")
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    test_results = generate_test(loaded_model, X_test, y_test)
    print("Test Results:")
    print(test_results)
    return test_results


def generate_predictions(url):
    print("Scraping the Data!")
    input_data = scrape_webpage(url)
    #input_data = input_data[input_data['votes'] >= 1]
    X_input = input_data[['text']]
    print("Done Scraping the data!")
    
    print("Start Sentiment Analysis!")
    loaded_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Generate Inputs\n")
    test_results = generate_input(loaded_model, X_input)

def run_training_pipeline():
    training_pipeline()

