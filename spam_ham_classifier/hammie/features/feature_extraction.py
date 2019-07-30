#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
==============================================================================
Classification of emails using sparse features
==============================================================================
This is a program to classify emails by topics using a bag-of-words approach. 
This program uses scikit-learn and a scipy.sparse matrix to store the features 
and demonstrates various classifiers that can efficiently handle sparse matrices.
"""

from __future__ import print_function
import csv
import logging
import numpy as np
from optparse import OptionParser
from os import listdir
from os.path import basename
from os.path import exists
from os.path import isdir
from os.path import join
import pickle
import re
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.model_selection import GridSearchCV
import sys
from time import time

import email
import re
import enchant

from email.parser import Parser
from keywords import abnormalkeywords
from sim_keywords import get_similar_words_count
from spambayes import mboxutils


###############################################################################
def usage():
    """Parse commandline arguments."""
    op = OptionParser()
    op.add_option("--train_path",
                  action="store", type=str,
                  help="Select train path.")
    op.add_option("--test_path",
                  action="store", type=str,
                  help="Select test path.")
    op.add_option("--email_charset",
                  action="store", type=str,
                  help="Select email_charset. Default charset is 'latin1'.")
    op.add_option("--chi2_select",
                  action="store", type="int", dest="select_chi2",
                  help="Select some number of features using a chi-squared test")
    op.add_option("--all_categories",
                  action="store_true", dest="all_categories",
                  help="Whether to use all categories or not.")
    op.add_option("--use_hashing",
                  action="store_true",
                  help="Use a hashing vectorizer.")
    op.add_option("--use_counting",
                  action="store_true",
                  help="Use a counting vectorizer.")
    op.add_option("--n_features",
                  action="store", type=int, default=2 ** 16,
                  help="n_features when using the hashing vectorizer.")
    op.add_option("--add_new_features",
                  action="store_true",
                  help="Extracting non-text features.")
    op.add_option("--print_matrix",
                  action="store_true",
                  help="Whether to print document-term matrix or not.")
    op.add_option("--term_matrix_file",
                  action="store", type=str, 
                  help="DOCUMENT_TERM_MATRIX filename when using print_matrix.")
    
    print(__doc__)
#     op.print_help()
#     print()
    return op

###############################################################################
class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

###############################################################################
class L1LinearSVC(LinearSVC):
    """ L1LinearSVC class declaration for ... """
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

###############################################################################
def get_email(text):
    endheader = text.find("\r\n\r\n")
    if endheader == -1:
        header = ''
        body = text
    else:
        header = text[:endheader]
        body = text[endheader + len("\r\n\r\n"):]
        
    return header, body

###############################################################################
def read_emails_from_disk(data_folder, categories=None, email_charset="latin1"):
    """Read emails from files in folders.
    Default email_charset="latin1", for CSDMC2010_SPAM email_charset="iso-8859-1"
    """
    filenames = []
    targets = []
    data = []
    
    # read data from emails
    folders = [f for f in sorted(listdir(data_folder))
                 if isdir(join(data_folder, f))]
    
    if categories is not None:
        folders = [f for f in folders if f in categories]
    else:
        categories = [f for f in folders]
        
    # encoding mess {I know nothing about the magic it does.}
    from cStringIO import StringIO
    from email.generator import Generator
    fp = StringIO()
    g = Generator(fp, mangle_from_=False, maxheaderlen=60)

    for folder in folders:
        folder_path = join(data_folder, folder)
        mbox = mboxutils.getmbox(folder_path)
        print("Reading emails from folder %s" % folder_path)
        for msg in mbox:
            g.flatten(msg)
            data.append(msg.as_string())
            # set targets here
            if folder[0].lower() == 'h':
                targets.append(0)       # ham
            else:
                targets.append(1)       # spam
    data = [d.decode(email_charset, 'strict') for d in data]
    
    return Bunch(data=data,
                 filenames=filenames,
                 categories=categories,
                 targets=targets,
                 DESCR='Data from E-mails')

###############################################################################
def get_abnormal_number(text):
    """Return a number of abnormal symbols in text"""
    abnormals = "#$%^&*/\\"
    return sum(text.count(l) for l in abnormals)

###############################################################################
def get_numbers_number(text):
    """Return a number of numbers in text"""
    numbers = "0123456789"
    return sum(text.count(l) for l in numbers)

###############################################################################
def get_punctuation_number(text):
    """Return a number of punctuation symbols in text"""
    punctuation = ".,?!-:;\"\'"
    return sum(text.count(l) for l in punctuation)

###############################################################################
def get_links_number(text):
    """Return a number of links and specific html tags in text"""
    links = ["http://", "https://", "<a>", "</a>", "href=",
             "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>",
             "<span>", "</span>", "<img"]
    return sum(text.lower().count(l) for l in links)

###############################################################################
def get_keywords_number(text):
    """Return a number of abnormal keywords in text. 
    Dependency: from keywords import abnormalkeywords
    """
    return sum(text.count(l) for l in abnormalkeywords)

###############################################################################
def get_unsubscribe_number(text):
    """Return a number of specific keywords in text"""
    unsubscribe = ["unsubscribe"]
    return sum(text.lower().count(l) for l in unsubscribe)

###############################################################################
def get_sender(text):
    search_str = 'From: '
    start = text.find(search_str)
    end = text.find('\n', start)
    if start == -1 or end == -1:
        return 'Empty sender'
    else:
        return text[start + len(search_str):end]

###############################################################################
def get_send_time(text):
    t = re.compile(r'[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}')
    iterator = t.finditer(text)
    if iterator:
        for match in iterator:
            h, m, s = match.group().split(':')
            
            # send time in 8:00:00 - 18:59:59
            if h is not None and 8 <= int(h) and int(h) <= 18:
                return 1
            else:
                return 0
            break

    return 0


##############################################################################
def get_subject(text):
    """ Returns the length of the Subject."""
    msg = Parser().parsestr(str(text))

    return msg.get("Subject", "")


##############################################################################
def get_subject_len(text):
    """ Returns the length of the Subject."""
    return len(get_subject(text))


##############################################################################
def get_miss_spell_count(text):
    """ Returns the number of mispelled words in text."""

    d = enchant.Dict("en_US")
    words = re.findall('[a-z]+', text.lower())
    miswords = filter(lambda word: not d.check(word), words)
    
    return len(miswords)

def get_simabnormal_count(text):
    """Returns the number of words similar to abnoral numbers in text."""
    subject = get_subject(text)
    return get_similar_words_count(subject, abnormalkeywords)

    
###############################################################################
def progress_bar(index, data_length):
    """Draw a progress bar in terminal"""
    bar_length = 60
    
    percent = float(index) / data_length
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\r[{0}] %{1} {2} {3}".format(hashes + spaces, str(int(round(percent * 100))),
                                                    str(index+1), str(data_length)))
    sys.stdout.flush()
    
###############################################################################
""" Dictionary of new features """
F_DICT = {0: {'func':len,                    'header':'h_len',         'body':'b_len',         'desc':'email header and body lengths'},  
          1: {'func':get_abnormal_number,    'header':'h_abnormal',    'body':'b_abnormal',    'desc':'number of abnormal symbols'},
          2: {'func':get_numbers_number,     'header':'h_numbers',     'body':'b_numbers',     'desc':'number of numbers symbols'},
          3: {'func':get_punctuation_number, 'header':'h_punctuation', 'body':'b_punctuation', 'desc':'number of punctuation symbols'},
          4: {'func':get_links_number,       'header':'h_links',       'body':'b_links',       'desc':'number of links symbols'},
          5: {'func':get_keywords_number,    'header':'h_keywords',    'body':'b_keywords',    'desc':'number of keywords'},
          6: {'func':get_unsubscribe_number, 'header':'',              'body':'b_unsubscribe', 'desc':'number of keyword "unsubscribe"'},
          7: {'func':get_send_time,          'header':'h_send_time',   'body':'',              'desc':'send time in 8:00:00 - 18:59:59'},
          8: {'func':get_subject_len,        'header':'h_subject_len', 'body':'',              'desc':'length of the subject field'},
          9: {'func':get_miss_spell_count,   'header':'h_mwords_num',  'body':'b_mwords_num',  'desc':'number of mis-spelled words'},
          10:{'func':get_simabnormal_count,  'header':'h_simabnormal', 'body':'b_simabnormal', 'desc':'similar to abnormal words.'} 
}

###############################################################################
def get_factors(data, progress = True):
    """Get E-mail factors"""
    data_len = len(data)
    factors = np.zeros((data_len, 2*len(F_DICT)), dtype=int)

    for i, message in enumerate(data):
        header, body = get_email(message)
        if header == None: header = ''
        if body == None: body = ''
        for j in F_DICT:
            func = F_DICT[j]['func']
            if F_DICT[j]['header'] != '':
                factors[i][2*j] = func(header)
            if F_DICT[j]['body'] != '':
                factors[i][2*j + 1] = func(body)
        if (progress):
            progress_bar(i, data_len)
    else:
        if (progress):
            print("\n")
    
    return factors

def print_factors(X_test):
    """ Print E-mail factors."""
    
    fcount = 2 * len(F_DICT)
    start_idx = 1000
    
    for i in xrange(fcount):
        if i % 2 == 0:
            print("%s: %.4f" % (F_DICT[i/2]['header'], X_test[0][start_idx + i]))
        else:
            print("%s: %.4f" % (F_DICT[(i-1)/2]['body'], X_test[0][start_idx + i]))
    print("\n")
        
    
    
###############################################################################
def size_mb(docs):
    """ Size of documents in MB """
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

###############################################################################
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

###############################################################################
def print_document_term_matrix(vectorizer, X_train, train_data,
                               dtm_file="document-term_matrix.txt"):
    """Print document-term matrix to dtm_file."""
    with open(dtm_file, "w") as f:
        vocab = np.zeros(len(vectorizer.vocabulary_), dtype='<U200')
        for word, index in vectorizer.vocabulary_.iteritems():
            vocab[index] = word
#             print('index:', index, 'word:', vocab[index])
        
        X_coo = X_train.tocoo()    
        M, N = X_coo.shape
        d = {}
        
        for findex in range(M):
            d[findex] = [ train_data.filenames[findex] ]
        
        for i in range(len(X_coo.row)):
            d[X_coo.row[i]].append( (vocab[X_coo.col[i]], str(X_coo.data[i])) )
            
        for findex in range(M):
            for i, item in enumerate(d[findex]):
                if i == 0:
                    f.write('\n' + item + '\n')
                else:
                    (word, value) = item
#                     print('word:', word, 'value:', value)
                    f.write('\t'.join([word.encode('utf-8'), value, '\n']))
                    
    print('Document-term matrix was saved into file %s\n' % dtm_file)
#     print('Document-term matrix was saved into file %s\n' % 'document-term_matrix.txt')

###############################################################################
def get_classifier(X_train, X_test, y_train, y_test):
    print('=' * 78)
    print("GradientBoostingClassifier")
    X_train = X_train.todense()
    X_test = X_test.todense()
    print('_' * 78)
    
    print("Training: ")
    clf = GradientBoostingClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    pred = clf.predict(X_train)
    score = metrics.f1_score(y_train, pred)
    print("train-f1-score:   %0.3f" % score)
    
    t0 = time()
    pred = clf.predict(X_test)     
    test_time = time() - t0
    
    test_score = np.empty(len(clf.estimators_))
    max_test_score = 0
    max_i = 0
    for i, pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = metrics.f1_score(y_test, pred)
        if test_score[i] > max_test_score:
            max_test_score = test_score[i]
            max_i = i
    
    #plt.plot(np.arange(len(clf.estimators_)) + 1, test_score, label='Test without new factors')
    print("test-f1-score:    %0.3f    stage: %d" % (max_test_score, max_i))
    print("test time:  %0.3fs" % test_time)
    
################################################################################
def get_classifier_for_save(X_train, y_train, clf_dump=None):
    print('=' * 78)
    print("GradientBoostingClassifier")
    X_train = X_train.todense()
    print('_' * 78)
    
    print("Training: ")
    clf = GradientBoostingClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    pred = clf.predict(X_train)
    score = metrics.f1_score(y_train, pred)
    print("train-f1-score:   %0.3f" % score)

    if clf_dump:
        with open(clf_dump, "w") as f:
            pickle.dump(clf, f)
            print("Clissifier was saved to dump")
    
##############################################################################
def get_new_features(train_data, test_data, X_train, X_test, y_train, y_test):
    """Extracting new features and add to X_train and X_test."""
    print("Getting factors for train data...")
    train_factors  = get_factors(train_data.data)
    
    print("Getting factors for test data...")
    test_factors  = get_factors(test_data.data)
                                       
    X_train = hstack([X_train, csr_matrix(train_factors)])
    X_test  = hstack([X_test, csr_matrix(test_factors)])
    
    # get senders for train
    senders = []
    for i, email in enumerate(train_data.data):
        senders.append(get_sender(email))
    senders = np.array(senders)
       
    vectorizer = CountVectorizer(ngram_range=(1, 1),analyzer='char_wb')
    print("CountVectorizer(ngram_range=(1, 1),analyzer='char_wb')")
    X_train_senders = vectorizer.fit_transform(senders)
    X_train = hstack([X_train, X_train_senders])
       
    # get senders for test
    senders = []
    for i, email in enumerate(test_data.data):
        senders.append(get_sender(email))
    senders = np.array(senders)
       
    X_test_senders = vectorizer.transform(senders)
    X_test = hstack([X_test, X_test_senders])
     
    print('=' * 78)
    print("GradientBoostingClassifier")
    X_train = X_train.todense()
    X_test = X_test.todense()
    print('_' * 78)
    
    print("Training: ")
    clf = GradientBoostingClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    pred = clf.predict(X_train)
    score = metrics.f1_score(y_train, pred)
    print("train-f1-score:   %0.3f" % score)
    
    t0 = time()
    pred = clf.predict(X_test)     
    
    test_score = np.empty(len(clf.estimators_))
    max_test_score = 0
    max_i = 0
    for i, pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = metrics.f1_score(y_test, pred)
        if test_score[i] > max_test_score:
            max_test_score = test_score[i]
            max_i = i
    
    test_time = time() - t0
    #plt.plot(np.arange(len(clf.estimators_)) + 1, test_score, label='Test with new factors')
    print("test-f1-score:    %0.3f    stage: %d" % (max_test_score, max_i))
    print("test time:  %0.3fs" % test_time)
    
##############################################################################
def get_new_features_results(train_data, test_data, 
                             X_train, X_test, y_train, y_test,
                             result_row, outfile="results.csv"
                             ):
    """Extracting new features and add to X_train and X_test for test results."""
    print("Getting factors for train data...")
    train_factors  = get_factors(train_data.data)
    
    print("Getting factors for test data...")
    test_factors  = get_factors(test_data.data)
                                       
    X_train = hstack([X_train, csr_matrix(train_factors)])
    X_test  = hstack([X_test, csr_matrix(test_factors)])
    
    print('=' * 78)
    print("GradientBoostingClassifier")
    X_train = X_train.todense()
    X_test = X_test.todense()
    print('_' * 78)
    
    print("Training: ")
    clf = GradientBoostingClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1,
                                     verbose = 1)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    pred = clf.predict(X_train)
    score = metrics.f1_score(y_train, pred)
    print("train-f1-score:   %0.3f" % score)
    
    t0 = time()
    pred = clf.predict(X_test)     
    test_score = metrics.f1_score(y_test, pred)
    test_time = time() - t0
    print("test-f1-score:    %0.3f" % test_score)
    print("test time:  %0.3fs" % test_time)
    
    # prepare data for results
    nham_right = nham_wrong = nspam_right = nspam_wrong = 0
    false_positives = []
    false_negatives = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            if pred[i] == 1:
                nham_right += 1
            else:
                nham_wrong += 1
                false_negatives.append(basename(test_data.filenames[i]))
        else:
            if pred[i] == 1:
                nspam_wrong += 1
                false_positives.append(basename(test_data.filenames[i]))
            else:
                nspam_right += 1

    # set part2 of result row
    result_row['NHAM RIGHT'] = nham_right 
    result_row['NHAM WRONG'] = nham_wrong
    result_row['FALSE POSITIVE RATE [%]'] = \
                float(nham_wrong) / float(result_row['NHAM TESTED']) * 100 
    result_row['TRUE POSITIVE RATE [%]'] = \
                float(nham_right) / float(result_row['NHAM TESTED']) * 100
    result_row['NSPAM RIGHT'] = nspam_right
    result_row['NSPAM WRONG'] = nspam_wrong
    result_row['FALSE NEGATIVE RATE [%]'] = \
                float(nspam_wrong) / float(result_row['NSPAM TESTED']) * 100
    result_row['NHAM UNSURE'] = 0
    result_row['HAM UNSURE RATE [%]'] = 0 
    result_row['NSPAM UNSURE'] = 0
    result_row['SPAM UNSURE RATE [%]'] = 0 
    result_row['TOTAL UNSURE RATE [%]'] = 0
    result_row['TRAIN MESS %'] = \
                float(result_row['NUM TEST MESS']) / \
                float(result_row['NUM TEST MESS'] + result_row['NUM TRAIN MESS']) * 100
    result_row['HAM CUTOFF'] = 1 
    result_row['SPAM CUTOFF'] = 1
    result_row['MAX DISCRIMINATORS'] = 0
    result_row['FALSE POSITIVES'] = ' '.join(false_positives) 
    result_row['FALSE NEGATIVES'] = ' '.join(false_negatives) 
    result_row['UNSURES'] = ''
    
    clf_dump = "clf.dmp"
    #with open(clf_dump, "w") as f:
    #    pickle.dump(clf, f)
    #    print("Clissifier was saved to dump")

    with open(outfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow((result_row['NHAM TESTED'], 
                        result_row['NHAM RIGHT'], 
                        result_row['NHAM WRONG'], 
                        result_row['FALSE POSITIVE RATE [%]'], 
                        result_row['TRUE POSITIVE RATE [%]'],
                        result_row['NSPAM TESTED'], 
                        result_row['NSPAM RIGHT'], 
                        result_row['NSPAM WRONG'], 
                        result_row['FALSE NEGATIVE RATE [%]'],
                        result_row['NHAM UNSURE'], 
                        result_row['HAM UNSURE RATE [%]'], 
                        result_row['NSPAM UNSURE'], 
                        result_row['SPAM UNSURE RATE [%]'], 
                        result_row['TOTAL UNSURE RATE [%]'],
                        result_row['TRAIN FOLDERS'], 
                        result_row['TEST FOLDERS'], 
                        result_row['NUM TRAIN MESS'], 
                        result_row['NUM TEST MESS'], 
                        result_row['TRAIN MESS %'],
                        result_row['HAM CUTOFF'], 
                        result_row['SPAM CUTOFF'], 
                        result_row['MAX DISCRIMINATORS'],
                        result_row['FALSE POSITIVES'], 
                        result_row['FALSE NEGATIVES'], 
                        result_row['UNSURES']
                        ))

###############################################################################
def get_new_features_for_save(train_data, X_train, y_train, clf_dump=None):
    """Extracting new features and add to X_train and X_test."""
    print("Getting factors for train data...")
    train_factors  = get_factors(train_data.data)
                                       
    X_train = hstack([X_train, csr_matrix(train_factors)])
    
    print('=' * 78)
    print("GradientBoostingClassifier")
    X_train = X_train.todense()
    print('_' * 78)
    
    print("Training: ")
    clf = GradientBoostingClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1)
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    pred = clf.predict(X_train)
    score = metrics.f1_score(y_train, pred)
    print("train-f1-score:   %0.3f" % score)

    if clf_dump:
        with open(clf_dump, "w") as f:
            pickle.dump(clf, f)
            print("Clissifier was saved to dump")

 ##############################################################################
def email_classifier(train_path="train", test_path="test"):
    # parse commandline arguments
    op = usage()
    (opts, args) = op.parse_args()
    
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    
    # get train_path
    if opts.train_path:
        train_path = opts.train_path
    
    # get test_path
    if opts.test_path:
        test_path = opts.test_path
    
    # load some categories from the training and testing sets
    if opts.all_categories:
        categories = None
    else: # default subdirectories in train directory and test directory
        categories = [
            'h', 's'
        ]
        
    print("Loading data for categories:")
    print(categories if categories else "all")
    
    # set email charset
    if opts.email_charset:
        email_charset = opts.email_charset
    else: # default charset
        email_charset = "latin1"
        
    # load data from disk
    train_data = read_emails_from_disk(train_path, categories, email_charset)
    test_data = read_emails_from_disk(test_path, categories, email_charset)
    categories = test_data.categories
    print('data loaded')
    
    # print data statistics after data loaded
    data_train_size_mb = size_mb(train_data.data) 
    data_test_size_mb = size_mb(test_data.data)
    
    print("%d documents - %0.3fMB (training set)" % (
        len(train_data.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(test_data.data), data_test_size_mb))
    print("%d categories" % len(categories))
    print()
    
    # split a training set and a test set
    y_train, y_test = train_data.targets, test_data.targets
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    
    # get vectorizer
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=opts.n_features)
        print("HashingVectorizer")
        X_train = vectorizer.transform(train_data.data)
    elif opts.use_counting:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        print("CountVectorizer(ngram_range=(1, 2))")
        X_train = vectorizer.fit_transform(train_data.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        print("TfidfVectorizer")
        X_train = vectorizer.fit_transform(train_data.data)
        
    # print extracting features statistics
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

 
    
    # print the document-term matrix -----------------------------------------------
    if opts.print_matrix and not opts.use_hashing:
        if opts.term_matrix_file:
            print_document_term_matrix(vectorizer, X_train, opts.term_matrix_file)
        else:
            print_document_term_matrix(vectorizer, X_train)
    # ------------------------------------------------------------------------------
    
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_data.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    
    # extracting best features by a chi-squared test
    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print("done in %fs" % (time() - t0))
        print()
    
    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())
    
    # get classifier results
    get_classifier(X_train, X_test, y_train, y_test)
    
    # extracting non-text features and add to X_train and X_test
    if opts.add_new_features:
        get_new_features(train_data, test_data, X_train, X_test, y_train, y_test)
        #plt.yticks(())
        #plt.legend(loc='best')
        #plt.show()
        
    print()
    print('=' * 78)

###############################################################################
def save_x_train_with_new_features(train_path="train", vectorizer_dump=None, clf_dump=None):
    # parse commandline arguments
    op = usage()
    (opts, args) = op.parse_args()
    
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    
    # get train_path
    if opts.train_path:
        train_path = opts.train_path
    
    # load some categories from the training and testing sets
    if opts.all_categories:
        categories = None
    else: # default subdirectories in train directory and test directory
        categories = [
            'h', 's'
        ]
        
    print("Loading data for categories:")
    print(categories if categories else "all")
    
    # set email charset
    if opts.email_charset:
        email_charset = opts.email_charset
    else: # default charset
        email_charset = "latin1"
        
    # load data from disk
    train_data = read_emails_from_disk(train_path, categories, email_charset)
    categories = train_data.categories
    print('data loaded')
    
    # print data statistics after data loaded
    data_train_size_mb = size_mb(train_data.data) 
    
    print("%d documents - %0.3fMB (training set)" % (
        len(train_data.data), data_train_size_mb))
    print("%d categories" % len(categories))
    print()

    # save a training set
    y_train = train_data.targets 

    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    
    # get vectorizer
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=opts.n_features)
        print("HashingVectorizer")
        X_train = vectorizer.transform(train_data.data)
    elif opts.use_counting:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        print("CountVectorizer(ngram_range=(1, 2))")
        X_train = vectorizer.fit_transform(train_data.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        print("TfidfVectorizer")
        X_train = vectorizer.fit_transform(train_data.data)
        
    # save vectorizer to dump
    if vectorizer_dump:
        with open(vectorizer_dump, "w") as f:
            pickle.dump(vectorizer, f)
            print("Vectorizer was saved to dump")

    # print extracting features statistics
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    # print the document-term matrix -----------------------------------------------
    if opts.print_matrix and not opts.use_hashing:
        if opts.term_matrix_file:
            print_document_term_matrix(vectorizer, X_train, train_data, opts.term_matrix_file)
        else:
            print_document_term_matrix(vectorizer, X_train, train_data)
    # ------------------------------------------------------------------------------
    
    # extracting best features by a chi-squared test
    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        print("done in %fs" % (time() - t0))
        print()
    
    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())
    
    # get classifier results and save to dump
    get_classifier_for_save(X_train, y_train)
    
    # extracting non-text features and add to X_train and X_test
    if opts.add_new_features:
        get_new_features_for_save(train_data, X_train, y_train, clf_dump)
        
    print()
    print('=' * 78)

###############################################################################
def email_classifier_results(train_path="train", test_path="test"):
    # parse command line arguments
    op = usage()
    (opts, args) = op.parse_args()
    
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    
    # get train_path
    if opts.train_path:
        train_path = opts.train_path
    
    # get test_path
    if opts.test_path:
        test_path = opts.test_path
    
    # load some categories from the training and testing sets
    if opts.all_categories:
        categories = None
    else: # default subdirectories in train directory and test directory
        categories = [
            'h', 's'
        ]
        
    print("Loading data for categories:")
    print(categories if categories else "all")
    
    # set email charset
    if opts.email_charset:
        email_charset = opts.email_charset
    else: # default charset
        email_charset = "latin1"
        
    # load data from disk
    train_data = read_emails_from_disk(train_path, categories, email_charset)
    test_data = read_emails_from_disk(test_path, categories, email_charset)
    categories = test_data.categories
    print('data loaded')
    
    # print data statistics after data loaded
    data_train_size_mb = size_mb(train_data.data) 
    data_test_size_mb = size_mb(test_data.data)
    
    print("%d documents - %0.3fMB (training set)" % (
        len(train_data.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(test_data.data), data_test_size_mb))
    print("%d categories" % len(categories))
    print()
    
    # split a training set and a test set
    y_train, y_test = train_data.targets, test_data.targets

    # set header to file of test results
    set_header_results()
    # set part1 of result row
    result_row = {
            'NHAM TESTED': sum(1 for i in range(len(y_test)) if y_test[i] == 1),
            'NSPAM TESTED': sum(1 for i in range(len(y_test)) if y_test[i] == 0),
            'TRAIN FOLDERS': ' '.join(categories),
            'TEST FOLDERS': ' '.join(categories),
            'NUM TRAIN MESS': len(y_train),
            'NUM TEST MESS': len(y_test),
        }
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    
    # get vectorizer
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=opts.n_features)
        print("HashingVectorizer")
        X_train = vectorizer.fit_transform(train_data.data)
    elif opts.use_counting:
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        print("CountVectorizer(ngram_range=(1, 2))")
        X_train = vectorizer.fit_transform(train_data.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        print("TfidfVectorizer")
        X_train = vectorizer.fit_transform(train_data.data)
        
    # print extracting features statistics
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()
    
    # print the document-term matrix -----------------------------------------------
    if opts.print_matrix and not opts.use_hashing:
        if opts.term_matrix_file:
            print_document_term_matrix(vectorizer, X_train, opts.term_matrix_file)
        else:
            print_document_term_matrix(vectorizer, X_train)
    # ------------------------------------------------------------------------------
    
    print("Extracting features from the test dataset using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(test_data.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()
    
    # extracting best features by a chi-squared test
    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        print("done in %fs" % (time() - t0))
        print()
    
    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = np.asarray(vectorizer.get_feature_names())
   
    
    # save vectorizer to dump
    vectorizer_dump = "vec.dmp"
    #if vectorizer_dump:
    #    with open(vectorizer_dump, "w") as f:
    #        pickle.dump(vectorizer, f)
    #        print("Vectorizer was saved to dump")

    # extracting non-text features and add to X_train and X_test
    get_new_features_results(train_data, test_data, 
                             X_train, X_test, y_train, y_test,
                             result_row
                             )
        
    print()
    print('=' * 78)

###############################################################################
def set_header_results(outfile="results.csv"):
    """Set header in csv file of results"""
    if not exists(outfile):
        result_header = ("NHAM TESTED", "NHAM RIGHT", "NHAM WRONG", 
                         "FALSE POSITIVE RATE [%]", "TRUE POSITIVE RATE [%]",
                         "NSPAM TESTED", "NSPAM RIGHT", "NSPAM WRONG", 
                         "FALSE NEGATIVE RATE [%]",
                         "NHAM UNSURE", "HAM UNSURE RATE [%]", "NSPAM UNSURE", 
                         "SPAM UNSURE RATE [%]", "TOTAL UNSURE RATE [%]",
                         "TRAIN FOLDERS", "TEST FOLDERS", 
                         "NUM TRAIN MESS", "NUM TEST MESS", "TRAIN MESS %",
                         "HAM CUTOFF", "SPAM CUTOFF", "MAX DISCRIMINATORS",
                         "FALSE POSITIVES", "FALSE NEGATIVES", "UNSURES")
        with open(outfile, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(result_header)

###############################################################################
if __name__ == "__main__":
    email_classifier_results()

    
