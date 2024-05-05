import numpy as np
import pandas as pd
import datacleaning
import warnings
warnings.filterwarnings("ignore")                     #忽略不必要的警告

train = pd.read_csv("D:\\QQ\\NLP-beginer\\task1\\dataset\\train.tsv",sep='\t')
test = pd.read_csv("D:\\QQ\\NLP-beginer\\task1\\dataset\\test.tsv",sep='\t')

from bs4 import BeautifulSoup
import  re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import BOW
import NG
import softmaxregression

test_clean=pd.DataFrame(datacleaning.clean(test))
test_clean.columns=['dataclean']
train_clean=pd.DataFrame(datacleaning.clean(train))
train_clean.columns=['dataclean']

train_df=pd.concat([train,train_clean],axis=1)
test_df=pd.concat([test,test_clean],axis=1)

train_x=train_df['dataclean']
train_y=train_df['Sentiment']

test_x=test_df['dataclean']

from sklearn.model_selection import StratifiedShuffleSplit

# 设置分层抽样的参数
n_splits = 1
test_size = 0.3
random_state = 42

# 创建 StratifiedShuffleSplit 对象
stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

# 进行分层抽样
for train_index, val_index in stratified_split.split(train_x,train_y):
    X_train, X_val = train_x[train_index], train_x[val_index]
    y_train, y_val = train_y[train_index], train_y[val_index]
    

bow=BOW.BagofWords()
df_xb=bow.transform(train_x)

ngram=NG.Ngram(3)
df_xn=ngram.N_Gram(train_x)
softmax_bow=softmaxregression.softmax_regression(5)
softmax_ngram=softmaxregression.softmax_regression(5)
softmax_bow.softmax(df_xb,y_train,X_val,y_val)




