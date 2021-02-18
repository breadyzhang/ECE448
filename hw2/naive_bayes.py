# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from collections import Counter
"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    spam = Counter()
    ham = Counter()
    spam_emails = 0
    spam_words = 0
    ham_emails = 0
    ham_words = 0
    out = []
    total_words = 0
    total_emails = len(train_set)
    # go through each email in the training set
    for s in range(len(train_set)):
        # go thorugh each word in the training set
        for word in train_set[s]:
            # check if the email belongs to spam or not spam (ham)
            if train_labels[s] == 1:
                ham[word] = ham[word] + 1
                ham_words = ham_words + 1
            else:
                spam[word] = spam[word] + 1
                spam_words = spam_words + 1
        # keep track of number of words total and number of spamemails / hamemails
        total_words = total_words+len(train_set[s])
        spam_emails = spam_emails+1 if train_labels[s] == 0 else spam_emails
        ham_emails = ham_emails+1 if train_labels[s] == 1 else ham_emails
    # time for dev set
    for email in dev_set:
        is_spam = math.log(spam_emails/total_emails)
        not_spam = math.log(ham_emails/total_emails)
        for word in email:
            # calculate likelihood of spam
            count = (spam[word] + smoothing_parameter) / (spam_words + smoothing_parameter*2)
            is_spam = is_spam + math.log(count)
            # calculate likelihood of ham
            count = (ham[word] + smoothing_parameter) / (spam_words+smoothing_parameter*2)
            not_spam = not_spam + math.log(count)
        # including pos_prior
        not_spam = not_spam + math.log(pos_prior)
        is_spam = is_spam + math.log(1-pos_prior)
        if is_spam > not_spam:
            # email is spam
            for word in email:
                spam[word] = spam[word] + 1
                spam_words = spam_words + 1
                total_words = total_words + 1
            out.append(0)
        else:
            for word in email:
                ham[word] = ham[word] + 1
                ham_words = ham_words + 1
                total_words = total_words + 1
            out.append(1)
    return out
