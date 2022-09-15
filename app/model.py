import pandas as pd
import tensorflow as tf

from analysis import analysis
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from utils import MODEL_DIR

def word_embeddings(data, y):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')
    encodings = tokenizer(data['text'].tolist(),
                          max_length=128,
                          padding=True,
                          truncation=True)
    encodings_df = tf.data.Dataset.from_tensor_slices(
        (dict(encodings), y['polarity'].values))
    return(encodings_df)


def model_fitting(X_train_encodings_df, X_val_encodings_df):
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    train_history = model.fit(x=X_train_encodings_df.shuffle(80000).batch(64),
          epochs=5,
          batch_size=64,
          validation_data=X_val_encodings_df.shuffle(80000).batch(64), verbose=1)
    print("Done training the model")

    print("Saving the model in the model directory")
    model.save_pretrained(MODEL_DIR)
    print("Done Saving the model in the model directory")
    return train_history


def generate_test(loaded_model, X_test, y_test):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    predictions = []
    i = 0
    for text in X_test['text'].tolist():
        text_tokens = tokenizer(text, max_length=128,
                                    padding=True,
                                    truncation=True
                                    )
        tf_output = loaded_model.predict(text_tokens['input_ids'])[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        result = tf.argmax(tf_prediction, axis=1)
        predictions.append(result.numpy()[0])
        print(i)
        i+=1
    res = classification_report(predictions, y_test['polarity'], output_dict=True)
    res_df = pd.DataFrame(res).transpose()
    return res_df


def generate_input(loaded_model, X_input):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    predictions = []
    i = 0
    for text in X_input['text'].tolist():
        text_tokens = tokenizer(text, max_length=128,
                                    padding=True,
                                    truncation=True
                                    )
        tf_output = loaded_model.predict(text_tokens['input_ids'])[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        result = tf.argmax(tf_prediction, axis=1)
        predictions.append(result.numpy()[0])
        i+=1
        
    
    # analysis
    predicted_data = pd.DataFrame({"text": X_input['text'], "predictions": predictions})
    print("Generating Sentiment Analysis\n")
    analysis(predicted_data)
    print("Done Generating Sentiment Analysis")

    return predictions
