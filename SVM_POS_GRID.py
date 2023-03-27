import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

svietimas_data = pd.read_excel(r'C:/Users/Pc/Desktop/STEMMED_POS.xlsx')

sentences = svietimas_data['text']
y = svietimas_data['sentiment']

vectorizer = TfidfVectorizer(lowercase=False)
vectorizer.fit(sentences)
sentences = vectorizer.transform(sentences)

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=42, stratify= y)

model = svm.SVC()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1, 10, 25]},
                    {'kernel': ['poly'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1, 10, 25]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 1, 10, 25]}
                    ]

grid_search = GridSearchCV(model, tuned_parameters, cv=10, scoring='accuracy', return_train_score=False)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)