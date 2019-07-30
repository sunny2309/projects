#! /usr/bin/env python

"""Usage: %(program)s [-D|-d] [options]

Where:
    -h
        show usage and exit
    -d FILE
        use the DBM store.  A DBM file is larger than the pickle and
        creating it is slower, but loading it is much faster,
        especially for large word databases.  Recommended for use with
        hammiefilter or any procmail-based filter.
        Default filename: %(DEFAULTDB)s
    -p FILE
        use the pickle store.  A pickle is smaller and faster to create,
        but much slower to load.  Recommended for use with sb_server and
        sb_xmlrpcserver.
        Default filename: %(DEFAULTDB)s
    -U
        Untrain instead of train.  The interpretation of -g and -s remains
        the same.
    -f
        run as a filter: read a single message from stdin, add a new
        header, and write it to stdout.  If you want to run from
        procmail, this is your option.
    -g PATH
        mbox or directory of known good messages (non-spam) to train on.
        Can be specified more than once, or use - for stdin.
    -s PATH
        mbox or directory of known spam messages to train on.
        Can be specified more than once, or use - for stdin.
    -u PATH
        mbox of unknown messages.  A ham/spam decision is reported for each.
        Can be specified more than once.
    -r
        reverse the meaning of the check (report ham instead of spam).
        Only meaningful with the -u option.
    -v
        vectorize.dmp file.
    -c
        clf.dmp file.
    -a PATH 
        mbox of test ham messages.
    -b PATH
        mbox of test spam messages.
"""

import sys
import os
import getopt
import cPickle as cpickle

from spambayes.Options import options, get_pathname_option
from spambayes import mboxutils, hammie, Corpus, storage
from classifiers.DocClassifier import DocClassifier
from features.feature_extraction import progress_bar

Corpus.Verbose = True

program = sys.argv[0] # For usage(); referenced by docstring above
clf = ""
vec = ""
charset = "latin1"

# encoding mess
reload(sys)
sys.setdefaultencoding('utf-8')

# Default database name
# This is a bit of a hack to counter the default for
# persistent_storage_file changing from ~/.hammiedb to hammie.db
# This will work unless a user had hammie.db as their value for
# persistent_storage_file
if options["Storage", "persistent_storage_file"] == \
   options.default("Storage", "persistent_storage_file"):
    options["Storage", "persistent_storage_file"] = \
                       os.path.join("~", ".hammiedb")
DEFAULTDB = get_pathname_option("Storage", "persistent_storage_file")

# Probability at which a message is considered spam
SPAM_THRESHOLD = options["Categorization", "spam_cutoff"] = 0.9
HAM_THRESHOLD = options["Categorization", "ham_cutoff"] = 0.15


# CART document classifier
doc_clf = None

def train(h, msgs, is_spam):
    """Train bayes with all messages from a mailbox."""
    mbox = mboxutils.getmbox(msgs)
    i = 0
    for msg in mbox:
        i += 1
        if i % 10 == 0:
            sys.stdout.write("\r%6d" % i)
            sys.stdout.flush()
        h.train(msg, is_spam)
    sys.stdout.write("\r%6d" % i)
    sys.stdout.flush()
    print

def untrain(h, msgs, is_spam):
    """Untrain bayes with all messages from a mailbox."""
    mbox = mboxutils.getmbox(msgs)
    i = 0
    for msg in mbox:
        i += 1
        if i % 10 == 0:
            sys.stdout.write("\r%6d" % i)
            sys.stdout.flush()
        h.untrain(msg, is_spam)
    sys.stdout.write("\r%6d" % i)
    sys.stdout.flush()
    print

def score(h, msgs, reverse=0):
    """Score (judge) all messages from a mailbox."""
    global doc_clf
    global charset

    from cStringIO import StringIO
    from email.generator import Generator
    fp = StringIO()
    g = Generator(fp, mangle_from_=False, maxheaderlen=60)

    # XXX The reporting needs work!
    mbox = mboxutils.getmbox(msgs)
    i = 0
    spams = hams = unsures = 0

    sys.stdout.write("Scoring now: ")
    for msg in mbox:
        i += 1

        sys.stdout.write("\r${0}".format(i))
        sys.stdout.flush()

        prob, clues = h.score(msg, True)
        if hasattr(msg, '_mh_msgno'):
            msgno = msg._mh_msgno
        else:
            msgno = i
        isspam = (prob >= SPAM_THRESHOLD)
        isham = (prob <= HAM_THRESHOLD)

        if isham:
            hams += 1
        else:
            g.flatten(msg)
            msg = msg.as_string()

            if doc_clf.predict(msg):
                spams += 1
            else:
                hams += 1
                 
    return (spams, hams, unsures)

def usage(code, msg=''):
    """Print usage message and sys.exit(code)."""
    if msg:
        print >> sys.stderr, msg
        print >> sys.stderr
    print >> sys.stderr, __doc__ % globals()
    sys.exit(code)

def main():
    """Main program; parse options and go."""
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hd:Ufg:s:p:u:r:v:c:a:b:e:')
    except getopt.error, msg:
        usage(2, msg)

    if not opts:
        usage(2, "No options given")
    
    global vec
    global clf
    global doc_clf
    global charset

    pck = DEFAULTDB
    good = []
    spam = []
    unknown = []
    tgood = []
    tspam = []
    reverse = 0
    untrain_mode = 0
    do_filter = False
    usedb = None
    mode = 'r'
    for opt, arg in opts:
        if opt == '-h':
            usage(0)
        elif opt == '-g':
            good.append(arg)
            mode = 'c'
        elif opt == '-s':
            spam.append(arg)
            mode = 'c'
        elif opt == "-f":
            do_filter = True
        elif opt == '-u':
            unknown.append(arg)
        elif opt == '-U':
            untrain_mode = 1
        elif opt == '-r':
            reverse = 1
        elif opt == '-v':
            vec = arg 
        elif opt == '-c':
            clf = arg
        elif opt == '-a':
            tgood.append(arg)
        elif opt == '-b':
            tspam.append(arg)
        elif opt == '-e':
            charset = arg
    
    pck, usedb = storage.database_type(opts)
    if args:
        usage(2, "Positional arguments not allowed")

    if usedb == None:
        usage(2, "Must specify one of -d or -D")

    save = False
   
    print("Spam Cutoff ", options["Categorization", "spam_cutoff"])
    print("Ham Cutoff ", options["Categorization", "ham_cutoff"])

    # get the training data directory for making a DocClassifier instance
    train_path = os.path.dirname(good[0])

    # let's train a DocClassifier beforehand
    doc_clf = DocClassifier(train_path, charset)
    doc_clf.train_cart()
    doc_clf.train_logreg() 
    doc_clf.train_knn()
    doc_clf.train_svm()

    h = hammie.open(pck, usedb, mode)

    if not untrain_mode:
        for g in good:
            print "Training ham (%s):" % g
            train(h, g, False)
            save = True

        for s in spam:
            print "Training spam (%s):" % s
            train(h, s, True)
            save = True
    else:
        for g in good:
            print "Untraining ham (%s):" % g
            untrain(h, g, False)
            save = True

        for s in spam:
            print "Untraining spam (%s):" % s
            untrain(h, s, True)
            save = True

    if save:
        h.store()

    if do_filter:
        msg = sys.stdin.read()
        filtered = h.filter(msg)
        sys.stdout.write(filtered)

    if unknown:
        spams = hams = unsures = 0
        for u in unknown:
            if len(unknown) > 1:
                print "Scoring", u
            s, g, u = score(h, u, reverse)
            spams += s
            hams += g
            unsures += u
        print "Total %d spam, %d ham, %d unsure" % (spams, hams, unsures)
 
    # testing now
    print ("Testing now..")
    test_path = os.path.dirname(tgood[0])
    test_pkl = test_path + '/test_features.pkl'
    
    if os.path.isfile(test_pkl):
        print("--> Found a test pickle file.")
        doc_clf.set_test_dict(test_pkl)
        
    spams = hams = unsures = 0
    fp = fn = 0
    if tgood:
        for u in tgood:
            if len(tgood) > 1:
                print "Scoring", u
            s, g, u = score(h, u, reverse)
            spams += s
            hams += g
            unsures += u
            fp += s 
    if tspam:
        for u in tspam:
            if len(tspam) > 1:
                print "Scoring", u
            s, g, u = score(h, u, reverse)
            spams += s
            hams += g
            unsures += u
            fn += g 
    total = spams + hams + unsures + 0.0
    print ("total=%d, hams=%d, spams=%d, unsures=%d" % (total, hams, spams, unsures))
    print ("fp count: %d, fn count: %d" % (fp, fn))
    print ("FP: %f%%, FN: %f%%" % ((fp/total * 100), (fn/total) * 100))
    print ()

    with open(test_pkl, "wb") as f:
        print("--> Pickling the test dict so far.")
        cpickle.dump(doc_clf.test_dict, f)

    
    with open("results.txt", "a") as f:
        f.write("=" * 78)
        f.write("\n")
        f.write("dataset: %s\n" % os.path.basename(train_path))
        f.write("-" * 78)
        f.write("\n")
        f.write("total=%d, hams=%d, spams=%d, unsures=%d\n" % (total, hams, spams, unsures))
        f.write("fp count: %d, fn count: %d\n" % (fp, fn))
        f.write("fp percentage: %f%%, fn percentage: %f%%\n" % ((fp/total * 100), (fn/total) * 100))
    
if __name__ == "__main__":
    main()
