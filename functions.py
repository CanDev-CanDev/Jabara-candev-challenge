#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:59:13 2018

@author: mariamjabara
"""

import nltk as nl
import re

def get_text(file):
    "Import text from a given file"
    
    f=open(file,'r')
    text=f.readlines()
    text=''.join(text) 
    return text

def remove_string_special_chars(s):
    "Changes any whitespace to one space"
    stripped=re.sub('\s+',' ',stripped)
    stripped=stripped.strip()
    return stripped

def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    
    return score

def count_words(sent):
    '''This function returns the total # of words in the input text'''
    
    count=0
    words=word_tokenize(sent)
    for word in words:
        count +=1
    return count

def create_freq_dict(sents):
    '''This function creates a frequency distribution dictionary for each word in each document'''
    
    i=0
    freqDict_list=[]
    for sent in sents:
        i += 1 
        freq_dict= {}
        words = word_tokensize(sent)
        for word in words:
            word=word.lower()
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
            temp={'doc_id':i,'freq_dict':freq_dict}
        freqDict_list.append(temp)
    return freqDict_list

def computeTF(doc_info,freqDict_list):
    '''tf=frequency of the term in the doc'''
    
    TF_scores = []
    for tempDict in freqDict_list:
        id = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id':id,'TF_score':tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],'key':k}
            TF_scores.append(temp)
            
    return TF_scores

def computeIDF(doc_info, freqDict_list):
    '''idf = ln(total number of docs/number of docs with term in it)
    '''
    IDF_scores = []
    counter = 0 
    for dict in freqDict_list:
        counter += 1
        for k in dict['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            temp = {'doc_id' : counter, 'IDF_score' : math.log(len(doc_info)/count), 'key' : k }
            
            IDF_scores.append(temp)
    
    return IDF_scores

def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = { 'doc_id' : j['doc_id'],'TFIDF_score' : j['IDF_score']*i['TF_score'],'key' :i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores

    
    
            
            
            

                



