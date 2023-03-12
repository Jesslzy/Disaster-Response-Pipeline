import sys
import sqlite3
import pickle
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
        
    """
    Load dataset from SQL database
    """
    
    # load data from database
    table = 'disaster_response'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table, engine)

    X = df['message'].astype(str).values
    y = df.drop(['message', 'original', 'genre', 'id'], axis = 1).values
    
    target_names = df.drop(['message', 'original', 'genre', 'id'], axis = 1).columns
    
    return X, y, target_names


def tokenize(text):
        
    """
    Clean messages for model training.
    """
    
    # Setting up lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Removing potential urls, punctuation and email sign
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    
    # Tocknize text
    text = word_tokenize(text)
    
    # Removing all non-english words and common English stopwords
    text = [w for w in text if w not in stopwords.words("english")]
        
    # Lemmatization
    clean_tokens = []
    for t in text:
        clean_tok = lemmatizer.lemmatize(t).lower().strip()
        clean_tok = lemmatizer.lemmatize(t, pos = 'v')
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
        
    """
    Building model to apply classification task.
    """
    
    pipeline = Pipeline([
        ('CV',CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf_multi',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
              'clf_multi__estimator__min_samples_split': [2, 3],
              'clf_multi__estimator__min_samples_leaf': [2, 3]

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, target_names):
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Produce clasification report
    for i in range(len(target_names)):
        print('Message Category: {} '.format(target_names[i]))
        print(classification_report(y_test[:, i], y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test[:, i], y_pred[:, i])))



def save_model(model, model_filepath):
    
    """
    Save trained model as a pickle file.
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, target_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, target_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()