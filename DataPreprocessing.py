import random
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import torch

def transform_to_index_tensor(pairs,rus_w2i,en_w2i,device):
    rus_tensor = []
    en_tensor = []

    for word in range(len(pairs[0])):
        en_tensor.append(en_w2i[pairs[0][word]])
    for word in range(len(pairs[1])):
        rus_tensor.append(rus_w2i[pairs[1][word]])

    return en_tensor,rus_tensor


def split_dataset(pairs,val_size=20.000,test_size=20.000):
    pairs = random.sample(pairs,len(pairs))
    pairs_test = pairs[:test_size]
    pairs_val = pairs[test_size:test_size+val_size]
    pairs_train = pairs[test_size+val_size:len(pairs)]

    return pairs_train,pairs_val,pairs_test

def custom_index_tokenizer(phrase,w2i,i2w):
    for word in phrase:
        if word in w2i:
            continue
        else:
            if bool(w2i) == False:
                w2i[word] = 0
                i2w[0] = word
            else:
                new_idx = list(i2w)[-1]+1
                w2i[word] = new_idx
                i2w[new_idx] = word
    return w2i,i2w

def filter_double_spaces(word_list):
    updated_word_list = []
    for w in range(len(word_list)):
        if not word_list[w] == '':
            updated_word_list.append(word_list[w])
    return updated_word_list

def get_data(filename='fra.txt',max_words=-1,plot_res=False):
    russian_word_list = []
    english_word_list = []

    #dictionaries for converting a word to a unique integer and the opposite
    russian_word_to_idx = {'<SOS>':0,'<EOS>':1,'<PAD>':2}
    english_word_to_idx = {'<SOS>':0,'<EOS>':1,'<PAD>':2}
    russian_idx_to_word = {0:'<SOS>',1:'<EOS>',2:'<PAD>'}
    english_idx_to_word = {0:'<SOS>',1:'<EOS>',2:'<PAD>'}

    pairs = []

    #read the dataset from the file
    with open(filename, "r", encoding="utf-8") as f:
        lines_list = f.read().split("\n")
        print("The file total translated words/phrases are: "+str(len(lines_list)))

    #get the phrases for each language to a different list
    word_counter = 0
    for i in range(len(lines_list)):
        if not max_words == -1:
            word_counter += 1
            if word_counter > max_words:
                break
        try:
            lines_list[i].split('\t')[1]
        except:
            continue
        russian_word_list.append(lines_list[i].split('\t')[1])
        english_word_list.append(lines_list[i].split('\t')[0])
    print("The total english phrases are: " + str(len(english_word_list)))
    print("The total russian phrases are: "+str(len(russian_word_list)))

    russian_lengths = []
    english_lengths = []
    russian_words_final = []
    english_words_final = []
    for phrase in range(len(russian_word_list)):
        #remove punc
        russian_words = re.sub(r'[^\w\s]', '', russian_word_list[phrase])
        english_words = re.sub(r'[^\w\s]', '', english_word_list[phrase])

        #to lower case
        russian_words = russian_words.lower()
        english_words = english_words.lower()


        russian_lengths.append(len(russian_words))
        english_lengths.append(len(english_words))

        #split to space
        russian_words = russian_words.split(' ')
        english_words = english_words.split(' ')

        #filter double spaces
        russian_words = filter_double_spaces(russian_words)
        english_words = filter_double_spaces(english_words)

        #add SOS and EOS tokens
        russian_words.insert(0, "<SOS>")
        russian_words.append("<EOS>")

        english_words.insert(0, "<SOS>")
        english_words.append("<EOS>")
        pairs.append([english_words,russian_words])

        russian_word_to_idx,russian_idx_to_word = custom_index_tokenizer(russian_words,russian_word_to_idx,russian_idx_to_word)
        english_word_to_idx, english_idx_to_word = custom_index_tokenizer(english_words, english_word_to_idx,english_idx_to_word)

        russian_words_final.append(russian_words)
        english_words_final.append(english_words)

    if plot_res:
        plt.hist(english_lengths, 15, alpha=0.5, label='English Lengths',edgecolor = "black")
        plt.hist(russian_lengths, 30, alpha=0.5, label='Russian Lengths',edgecolor = "black")
        plt.legend(loc='upper right')
        plt.show()

    print("Found "+str(len(list(russian_idx_to_word.keys())))+" unique russian words.")
    print("Found " + str(len(list(english_idx_to_word.keys()))) + " unique english words.")

    return russian_words_final,english_words_final,russian_word_to_idx,russian_idx_to_word,english_word_to_idx,english_idx_to_word,pairs,russian_words_final,english_words_final