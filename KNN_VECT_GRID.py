import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

svietimas_data = pd.read_excel(r'C:/Users/Pc/Desktop/STEMMED_POS.xlsx')

sentences = svietimas_data['text']
y = svietimas_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=42, stratify= y)

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', KNeighborsClassifier(n_neighbors=29, weights='distance', metric='euclidean')),
])
parameters = [{
    'vect__max_df': (100,150,300, 1.0),
    'vect__min_df': (1,10,15),
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
},
    {
        'vect': [CountVectorizer()],
        'vect__max_df': (100,150,300),
        'vect__min_df': (1,10,15),
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    }]

grid_search = GridSearchCV(pipeline, parameters)

grid_search_tune = GridSearchCV(pipeline, parameters, cv=10, n_jobs=2, verbose=3)
grid_search_tune.fit(X_train, y_train)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)