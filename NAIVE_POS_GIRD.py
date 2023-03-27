import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
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

vectorizer = TfidfVectorizer(lowercase=False, min_df=10)
vectorizer.fit(sentences)
sentences = vectorizer.transform(sentences)

classifier = MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=42, stratify= y)

tuned_parameters = [{'alpha': (1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001)}]

grid_search = GridSearchCV(classifier, tuned_parameters, cv=10, scoring='accuracy', return_train_score=False)
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)