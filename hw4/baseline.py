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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    out = []
    words = {} # (freq of tag, tag)
    word_tag = {}
    tags = Counter()
    # tags = {} # (freq of word, word)
    # training
    for message in train:
        for word in message:
            if word[0] not in words:
                words[word[0]] = [word[1]]
                word_tag[(word[0],word[1])] = 1
            elif word[1] not in words[word[0]]:
                words[word[0]].append(word[1])
                word_tag[(word[0],word[1])] = 1
            else:
                word_tag[(word[0],word[1])] += 1
            tags[word[1]] += 1
        # if word[1] not in tags:
        #     tags[word[1]] = (1, word[0])
        # else:
        #     tags[word[1]] = (tags[word[1]][0] + 1, word[0])

    # testing
    for message in test:
        message_out = []
        for word in message:
            best_tag = ""
            value = 0
            if word in words:
                for tag in words[word]:
                    if word_tag[(word,tag)] > value:
                        value = word_tag[(word,tag)]
                        best_tag = tag
            else:
                best_tag = max(tags)
            message_out.append((word,best_tag))
        out.append(message_out)
    return out
