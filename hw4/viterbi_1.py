# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from collections import Counter
import math
def laplace(xy,y,x,k):
    smoothing = math.log((xy+k)/(y+k*(x+1)))
    return smoothing

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    out = []
    smoothing = 0.00005
    unique_tags = 0
    unique_words = 0
    word_counter = Counter()
    tag_counter = Counter()
    word_tag = {} # word:{tag:freq}
    tag_tag = {} # prev_tag:{tag:freq}
    emission = {} # tag: {word:prob}
    transition = {} # prev_tag:{tag:prob}
    # get necessary data for setting up emission and transition
    for message in train:
        for i in range(len(message)-1):
            word = message[i][0]
            tag = message[i][1]
            prev_tag = message[i-1][1]
            if prev_tag not in tag_tag:
                tag_tag[prev_tag] = Counter()
            tag_tag[prev_tag][tag] += 1
            if tag not in word_tag:
                word_tag[tag] = Counter()
            word_tag[tag][word] += 1
            word_counter[word] += 1
            tag_counter[tag] += 1
    unique_tags = len(tag_counter) - 1 # take off 'S'
    unique_words = len(word_counter)
    # print("words:",word_tag)
    # print("tag: ",tag_tag)
    # print(unique_tags,unique_words)
    # setting up emission
    for tag in word_tag:
        emission[tag] = {}
        for word in word_tag[tag]:
            emission[tag][word] = laplace(word_tag[tag][word],tag_counter[tag],unique_words,smoothing)
    #setting up transition
    for prev in tag_tag:
        transition[prev] = {}
        for tag in emission:
            if tag in tag_tag[prev]:
                transition[prev][tag] = laplace(tag_tag[prev][tag],tag_counter[prev],unique_words,smoothing)
            else:
                transition[prev][tag] = laplace(0,tag_counter[prev],unique_words,smoothing)
    # print("emission: ",emission["START"])
    # testing
    for message in test:
        backtrack = {}
        probs = {}
        prediction = []
        for tag in emission:
            probs[(0,tag)] = transition["START"][tag] #+ emission[tag][message[0]]
            backtrack[(0,"START")] = 0
        for i in range(1,len(message)):
            word = message[i]
            for tag in emission:
                rates = float("-inf")
                best_tag = ""
                for prev in transition[tag]:
                    likelihood = 0
                    if word in emission[tag]:
                        likelihood = transition[prev][tag] + probs[(i-1,prev)] + emission[tag][word]
                    else:
                        likelihood = transition[prev][tag] + probs[(i-1,prev)] + laplace(0,tag_counter[tag],unique_words,smoothing)
                    if rates < likelihood:
                        rates = likelihood
                        best_tag = prev
                backtrack[(i,tag)] = (i-1,best_tag)
                probs[(i,tag)] = rates
        most_likely_tag = ""
        rate = float("-inf")
        for tag in emission:
            if rate < probs[len(message)-1,tag]:
                most_likely_tag = tag
                rate = probs[len(message)-1,tag]
        i = len(message)-1
        tuple = (i,most_likely_tag)
        while i > 0:
            prediction.insert(0,(message[i],tuple[1]))
            tuple = backtrack[tuple]
            i -= 1
        prediction.insert(0,("START","START"))
        out.append(prediction)
    print(out[0])
    return out
