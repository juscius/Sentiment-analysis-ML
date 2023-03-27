import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

svietimas_data = pd.read_excel(r'C:/Users/Pc/Desktop/STEMMED_POS.xlsx')

sentences = svietimas_data['text']
y = svietimas_data['sentiment']

vectorizer = TfidfVectorizer(lowercase=False)
vectorizer.fit(sentences)
sentences = vectorizer.transform(sentences)

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=42, stratify= y)

classifier = KNeighborsClassifier()

weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range, weights = weights, metric = metric )

grid_search = GridSearchCV(classifier, param_grid, cv=10,scoring='accuracy', return_train_score=False)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
