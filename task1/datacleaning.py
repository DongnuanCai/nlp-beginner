from bs4 import BeautifulSoup
import  re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

def clean(df):
    afterprocess = []
    lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for Phra in df['Phrase']:
        process = BeautifulSoup(Phra, 'html.parser').get_text()
        process = re.sub("[^a-zA-Z\s]", '', process)  # 过滤除英文字符以外数据
        process = process.lower()  # 小写化
        word = word_tokenize(process)
        process = [lem.lemmatize(w) for w in word]#if w not in stop_words]  # 词形还原并去除停用词
        afterprocess.append(' '.join(process))

    return afterprocess






