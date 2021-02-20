# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
import math
from collections import Counter
"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set
    # (1-lambda)log(P(Y)) + sum( logP(wi|Y)) + lambda * logP(Y) + sum(logP(bi|Y))
    unigram_spam = Counter()
    bigram_spam = Counter()
    unigram_ham = Counter()
    bigram_ham = Counter()
    spam_words = 0
    bigram_spam_words = 0
    ham_words = 0
    bigram_ham_words = 0
    out = []
    total_emails = len(train_set)
    # go through each email in the training set
    for s in range(len(train_set)):
        # go thorugh each word in the training set
        for i in range(len(train_set[s])):
            if train_labels[s] == 0:
                # add bigram tuple and unigram word to dicts
                if i < len(train_set[s])-1:
                    bigram_spam[(train_set[s][i], train_set[s][i+1])] += 1
                    bigram_spam_words += 1
                unigram_spam[train_set[s][i]] += 1
                spam_words += 1
            else:
                if i < len(train_set[s])-1:
                    bigram_ham[(train_set[s][i], train_set[s][i+1])] += 1
                    bigram_ham_words += 1
                unigram_ham[train_set[s][i]] += 1
                ham_words += 1
    # time for dev set
    for email in dev_set:
        spam_unigram = math.log(1-pos_prior)
        spam_bigram = math.log(1-pos_prior)
        ham_unigram = math.log(pos_prior)
        ham_bigram = math.log(pos_prior)
        for w in range(len(email)):
            # calculate unigram likelihood of spam
            unigram = (unigram_spam[email[w]]+unigram_smoothing_parameter) / (spam_words + unigram_smoothing_parameter*len(unigram_spam))
            spam_unigram += math.log(unigram)
            # calculate unigram likelihood of ham
            unigram = (unigram_ham[email[w]]+unigram_smoothing_parameter) / (ham_words + unigram_smoothing_parameter*len(unigram_ham))
            ham_unigram += math.log(unigram)
            if w < len(email)-1:
                # calculate likelihood of spam bigram
                bigram = (max(bigram_spam[(email[w], email[w+1])], bigram_spam[(email[w+1],email[w])])+bigram_smoothing_parameter) / (bigram_spam_words + bigram_smoothing_parameter*len(bigram_spam))
                spam_bigram += math.log(bigram)
                # calculate likelihood of ham bigram
                bigram = (max(bigram_ham[(email[w], email[w+1])], bigram_ham[(email[w+1],email[w])])+bigram_smoothing_parameter) / (bigram_ham_words + bigram_smoothing_parameter*len(bigram_ham))
                ham_bigram += math.log(bigram)
        is_spam = spam_unigram*(1-bigram_lambda)+spam_bigram*bigram_lambda
        is_ham = ham_unigram*(1-bigram_lambda) + ham_bigram*bigram_lambda
        if is_spam > is_ham:
            out.append(0)
        else:
            out.append(1)
    return out
