#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:32:02 2018

@author: mariamjabara
"""

import nltk as nl
import functions
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk import trigrams
from textblob import Word 
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PIL

#get_text('raw_data_copy.txt')

with open('raw_data3.txt','r') as myfile:
   data=myfile.readlines()

#manipulating the data
del data[0]

#lowercase
for i in range(0,2000):
   data[i]=data[i].lower()

#remove punctuation:
data = [''.join(c for c in s if c not in string.punctuation) for s in data]

for i in range(0,2000):
   data[i]=data[i].lstrip()
   data[i]=data[i].rstrip()
   
#removing truncations:
for index, item in enumerate(data):
    if item == '':
        del data[index]
        
   
stop_words=set(stopwords.words('english'))
word_tokens=[]
a=len(data)
stop_filtered = [[] for _ in range(a)]
tokenized = [[] for _ in range(a)]
for i in range(0,a):
   item=word_tokenize(data[i])
   tokenized[i]=item
#   for w in item:
#       if w not in stop_words:
#           stop_filtered[i].append(w)

#lemmatizing
lemmatized = [[] for _ in range(a)]
for i in range(0,a):
    for j in tokenized[i]:
        w=Word(j)
        lem=w.lemmatize()
        lemmatized[i].append(lem)
        
#bigrams:
bi_grams = [[] for _ in range(a)]
for i in range(0,a):
    if len(lemmatized[i])==1:
        pass
    else:
        bi_grams[i]=list(bigrams(lemmatized[i]))    

tri_grams = [[] for _ in range(a)]
for i in range(0,a):
    if len(lemmatized[i])==1 or len(lemmatized[i])==2:
        pass
    else:
        tri_grams[i] = list(trigrams(lemmatized[i]))

master_list=[]        
for i in range (0,1983):
    master_list.append(' '.join(lemmatized[i]))

master_data=''
for i in range (0,1983):
    master_data+=master_list[i]

nltk_results=[nltk_sentiment(i) for i in data]
results_df=pd.DataFrame(nltk_results)
text_df=pd.DataFrame(data,columns=['text'])
nltk_df=text_df.join(results_df)

basecloud=WordCloud().generate(master_data)
image = basecloud.to_image()
image.show()


