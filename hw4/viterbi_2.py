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
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import Counter
import math
def laplace(xy,y,x,k):
    smoothing = math.log((xy+k)/(y+k*(x+1)))
    return smoothing

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    out = []
    smoothing = 0.01
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
    hapax = []
    hapax_tag = Counter()
    hapax_smoothing = {}
    for word in word_counter:
        if word_counter[word] == 1:
            hapax.append(word)
    # print("words:",word_tag)
    # print("tag: ",tag_tag)
    # print(unique_tags,unique_words)
    # print(hapax)
    # setting up hapax
    for tag in tag_counter:
        for word in word_tag[tag]:
            if word in hapax:
                hapax_tag[tag] += 1
    for tag in tag_counter:
        hapax_smoothing[tag] = (hapax_tag[tag]+smoothing)/(len(hapax)+smoothing*(tag_counter[tag]+1))
    # print(hapax_smoothing)
    # print(hapax_tag)
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
                transition[prev][tag] = laplace(tag_tag[prev][tag],tag_counter[prev],unique_words,hapax_smoothing[tag])
            else:
                transition[prev][tag] = laplace(0,tag_counter[prev],unique_words,hapax_smoothing[tag])
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
                        likelihood = transition[prev][tag] + probs[(i-1,prev)] + laplace(0,tag_counter[tag],unique_words,hapax_smoothing[tag])
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
    return out
