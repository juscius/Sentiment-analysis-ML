import numpy as np
import pandas as pd
import re
from sklearn import svm
import pandas

svietimas_data = pd.read_excel(r'C:/Users/Pc/Desktop/NEG_BALANCE.xlsx')

my_file = open("stopwords.txt", "r", encoding="utf8")
data = my_file.read()
stop_words_list = data.split('\n')
my_file.close()

def clean_data(data):
    data = str(data).lower()
    data = re.sub("\n+", '', data)
    letters = re.sub(r'[0-9]', '', data)
    punctuation = re.sub(r'[^\w\s]', '', letters)
    return punctuation

def stop_words(words):
    filter_words = []
    for word in words:
        if word not in stop_words_list:
            filter_words.append(word)
    return filter_words

def words_length(words):
    filter_word = []
    for word in words:
        if len(word) > 2:
            filter_word.append(word)
    return filter_word

svietimas_data['text'] = svietimas_data['text'].apply(lambda x: x.split(' '))
svietimas_data['text'] = svietimas_data['text'].apply(lambda x: clean_data(x))
svietimas_data['text'] = svietimas_data['text'].apply(lambda x: x.split(' '))
svietimas_data['text'] = svietimas_data['text'].apply(lambda x: stop_words(x))
svietimas_data['text'] = svietimas_data['text'].apply(lambda x: words_length(x))
svietimas_data['text'] = svietimas_data['text'].apply(lambda x: ' '.join(x))

df = pd.DataFrame(svietimas_data)
df.to_excel('NEG.xlsx', encoding='utf_8_sig')