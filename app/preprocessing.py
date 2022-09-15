import nltk
import pandas as pd
import re
import string

from nltk.corpus import stopwords


def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def preprocessing(train_data):

    # add title with review
    train_data['text'] = train_data['title'].astype(
        str) + " " + train_data["review"]
    train_data = train_data.iloc[:, [0, -1]]

    # remove emojis and special characters
    train_data['text'] = train_data['text'].str.replace(
        '[^A-Za-z0-9\s+]', '', flags=re.UNICODE)
    train_data['text'] = train_data['text'].apply(lambda x: x.lower())
    train_data['text'] = train_data['text'].apply(
        lambda x: remove_punctuation(x))
    train_data['text'] = train_data['text'].str.replace(r"@", " at ")
    train_data['text'] = train_data['text'].str.replace(
        "#[^a-zA-Z0-9_]+", " ")  # no hashtag
    train_data['text'] = train_data['text'].str.replace(
        r"[^a-zA-Z(),\"'\n_]", " ")
    train_data['text'] = train_data['text'].str.replace(r"http\S+", "")

    stop_words = stopwords.words('english')
    train_data['text'] = train_data['text'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop_words)]))
    train_data['text'] = train_data['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word.isalpha()]))

    # Lemmatization
    train_data['text'] = train_data['text'].apply(lemmatize_text)
    return train_data


def ratings_to_label(data):
    if data > 2:
        return 1
    else:
        return 0


def preprocessing_input(input_data):

    # add title with review
    input_data['text'] = input_data['title'].astype(
        str) + " " + input_data["review"]
    input_data = input_data.iloc[:, 2:]

    # remove emojis and special characters
    input_data['text'] = input_data['text'].str.replace(
        '[^A-Za-z0-9\s+]', '', flags=re.UNICODE)
    input_data['text'] = input_data['text'].apply(lambda x: x.lower())
    input_data['text'] = input_data['text'].apply(
        lambda x: remove_punctuation(x))
    input_data['text'] = input_data['text'].str.replace(r"@", " at ")
    input_data['text'] = input_data['text'].str.replace(
        "#[^a-zA-Z0-9_]+", " ")  # no hashtag
    input_data['text'] = input_data['text'].str.replace(
        r"[^a-zA-Z(),\"'\n_]", " ")
    input_data['text'] = input_data['text'].str.replace(r"http\S+", "")

    stop_words = stopwords.words('english')
    input_data['text'] = input_data['text'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stop_words)]))
    input_data['text'] = input_data['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word.isalpha()]))

    # Lemmatization
    input_data['text'] = input_data['text'].apply(lemmatize_text)

    return input_data
