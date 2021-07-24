import numpy as np
from random import shuffle
from nltk.stem import WordNetLemmatizer
import re


def train_test_val():
    np.random.seed(42)
    
    messages = open('../messages.txt').read().split('\n')
    shuffle(messages)
    
    X = []
    y = []
    for text in messages:
        split = text.split('\t', 1)
        if len(split) == 2 and len(split[1]) > 0:
            X.append(standardization(split[1])) #standardize for word embeddings
            y.append(split[0])
            
    y = [1 if label == '[Sydney]' else 0 for label in y]

    X_train, X_test, X_val = np.split(X, [int(.6*len(X)), int(.8*len(X))])
    y_train, y_test, y_val = np.split(y, [int(.6*len(y)), int(.8*len(y))])

    return X_train, y_train, X_test, y_test, X_val, y_val


def standardization(data):
    lemmatizer = WordNetLemmatizer()
    
    custom = data.lower()
    custom = re.sub(r'[^ ]+\.[^ ]+', '[LINK]', custom)
    custom = re.sub(r'[^a-zA-Z0-9_ ]', '', custom)
    custom = lemmatizer.lemmatize(custom)
    
    return custom