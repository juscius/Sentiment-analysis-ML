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

i = 5
accuracyList = []
for time in range(5):

    X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=i, stratify= y)
    classifier = KNeighborsClassifier(n_neighbors=9, metric='euclidean', weights='distance')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracyList.append(accuracy_score(y_test, y_pred))
    i = i+1

print(sum(accuracyList)/len(accuracyList))
print(accuracyList)