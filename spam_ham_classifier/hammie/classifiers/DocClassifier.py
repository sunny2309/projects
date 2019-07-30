#!/usr/bin/python
# -*- coding: utf-8 -*-

import cPickle as cpickle
import os
import hashlib
import warnings

warnings.filterwarnings('ignore')

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from features.feature_extraction import *


class DocClassifier():
    """A class to keep state and trained model."""

    def __init__(self, train_path = "train", charset = 'latin1'):
        """ Initialize the classifier. Set associated data and vectorizers."""

        # Vectorizers
        self.vec = None
        self.ch2 = None

        # Classifiers
        self.clf = None
        self.svm = None
        self.logreg = None
        self.knn = None
        
        # Data
        self.train_path = train_path
        self.charset = charset
        self.feature_names = None
        self.features_pkl = train_path + "/features.pkl"
        self.test_dict_p = False
        self.test_dict = {}
        
        # If pickle file exists then load things directly
        if os.path.isfile(self.features_pkl):
            print("--> Found a pickle file: %s" % self.features_pkl)
            with open(self.features_pkl, "rb") as f:
                self.vec = cpickle.load(f)
                self.ch2 = cpickle.load(f)
                self.X = cpickle.load(f)
                self.y = cpickle.load(f)
            return 

        categories = ['h', 's']
        
        print("Loading data from subfolders.")
    
        # load data from disk
        train_data = read_emails_from_disk(self.train_path, categories, self.charset)
        categories = train_data.categories
        print('data loaded')
    
        # print data statistics after data loaded
        data_train_size_mb = size_mb(train_data.data) 
    
        print("%d documents - %0.3fMB (training set)" % (
            len(train_data.data), data_train_size_mb))
        print("%d categories" % len(categories))
    
        y_train = train_data.targets

        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()

        # use count vectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        print("CountVectorizer(ngram_range=(1, 2))")
        X_train = vectorizer.fit_transform(train_data.data)
        
        # print extracting features statistics
        duration = time() - t0
        print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
        print("n_samples: %d, n_features: %d" % X_train.shape)

        # extracting best features by a chi-squared test
        select_chi2 = 1000
        print("Extracting %d best features by a chi-squared test", select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=select_chi2)

        print ("X_train shape before chi-square: ", X_train.shape)
        X_train = ch2.fit_transform(X_train, y_train)
        print ("X_train shape after chi-square: ", X_train.shape)
        print("done in %fs" % (time() - t0))

        # save this chi2 vectorizer and feature_names
        self.ch2 = ch2
        self.feature_names = np.asarray(vectorizer.get_feature_names())[ch2.get_support()]

        # store the vectorizer state now
        self.vec = vectorizer
        
        print("Getting factors for train data...")
        train_factors  = get_factors(train_data.data)
        print("train_factors: ", train_factors.shape) 

        X_train = hstack([X_train, csr_matrix(train_factors)])
        X_train = X_train.todense()

        self.X = X_train
        self.y = y_train
        
        with open(self.features_pkl, "wb") as f:
            cpickle.dump(self.vec, f)
            cpickle.dump(self.ch2, f)
            cpickle.dump(self.X, f)
            cpickle.dump(self.y, f)

    def set_test_dict(self, test_pkl):
        """Extracts test dict from the test_pkl file."""

        with open(test_pkl, "rb") as f:
            self.test_dict = cpickle.load(f)
            self.test_dict_p = True
            
    def get_features(self, msg):
        """ Get the features for a msg after the training has been done once."""
        
        msg = msg.decode(self.charset, 'strict')
                
        vectorizer = self.vec
        ch2 = self.ch2  
	msgh = hashlib.sha224(msg).hexdigest() 

        if (self.test_dict_p) and self.test_dict.has_key(msgh):
            return self.test_dict[msgh]
        else:
            # get X_test for msg
            X = vectorizer.transform([msg])
        
            # select best select_chi2 features
            select_chi2 = 1000
            X = ch2.transform(X)

            factors  = get_factors([msg], progress = False)

            # add factors to X_test
            X = hstack([X, csr_matrix(factors)])
            X = X.todense()
            self.test_dict[msgh] = X

            return X
        
    def predict_cart(self, msg):
        """Check unsure msg for spam or ham and return True if spam and False if ham."""
        X_test = self.get_features(msg)            
        pred = self.clf.predict(X_test)

        if pred == 1:
            return True     # spam
        else:
            return False    # ham

    def train_cart(self):
        """ Train the classifier."""

        print('=' * 78)
        print("GradientBoostingClassifier")
        print('_' * 78)

        X_train = self.X
        y_train = self.y
        
        print("Training: ")
        clf = GradientBoostingClassifier(n_estimators=200,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         verbose = 1)
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        
        print ("X_train shape:", X_train.shape)

        pred = clf.predict(X_train)
        score = metrics.f1_score(y_train, pred)
        print("train-f1-score:   %0.3f" % score)

        # save classifier state now
        self.clf = clf
        
        print()
        print("Training done.")
        print('=' * 78)

    def gridsearch_cart(self):
        """Search for the optimal parameters for GradientBoostingClassifier."""

        X_train = self.X
        y_train = self.y
        
        param_grid = { 'learning_rate': [0.1, 0.05, 0.02, 0.01]
                     , 'max_depth': [3, 4, 6]
                     , 'min_samples_leaf': [1, 3, 5, 9, 17]
                     }

        est = GradientBoostingClassifier(n_estimators=200)
        gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_train, y_train)

        # best hyperparameter setting
        print("Best hyper-parameter settings: ")
        print (gs_cv.best_params_)

    def train_svm(self):
        """ Train an SVM for doing classification."""

        self.svm = svm.LinearSVC() # max_iter=5000000

        print("--> Training an SVM on the dataset.")
        self.svm.fit(self.X, self.y)
        print("--> Done!")

    def predict_svm(self, msg):
        """ Ask the trained svm for classifying an email."""
        X_test = self.get_features(msg)
        pred = self.svm.predict(X_test)

        if pred == 1:
            return True 
        else:
            return False

    def train_logreg(self):
        """Train a logistic regression model."""
        est = LogisticRegression() # solver='lbfgs',max_iter=10000

        print("--> Training a logistic regression model.")
        est.fit(self.X, self.y)
        print("--> Done!")

        self.logreg = est

    def predict_logreg(self, msg):
        """Predict spamminess with logistic regression."""
        pred = self.logreg.predict(self.get_features(msg))

        if pred == 1:
            return True
        else:
            return False

    def train_knn(self):
        """Train a K-nearest neihbor estimator on data."""
        est = KNeighborsClassifier(n_neighbors=1)
        print("--> Training a KNeihborsClassifer estimator.")
        est.fit(self.X, self.y)
        print("--> Done!")

        self.knn = est
        
    def predict_knn(self, msg):
        """Predict class label with KNN."""
        pred = self.knn.predict(self.get_features(msg))

        if pred == 1:
            return True
        else:
            return False

    def predict(self, msg):
        """ The exported overall predict function that uses the other predict 
        functions.
        """
        predictions = self.predictions(msg)
        positives = predictions.count(True)
        negatives = predictions.count(False)

        if positives > negatives:
            return True
        else:
            return False

    def predictions(self, msg):
        """Returns the predictions for all the classifiers we have."""
        X_test = self.get_features(msg)

        allpredictions = [ self.predict_knn(msg)
			, self.predict_cart(msg)
			, self.predict_svm(msg)
			, self.predict_logreg(msg) ]

        return allpredictions

###############################################################################
if __name__ == "__main__":
    print ("TODO: Code for testing the module")
    
