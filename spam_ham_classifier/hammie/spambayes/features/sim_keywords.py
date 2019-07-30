#!/usr/bin/python2
# -*- coding: utf-8 -*-
#===============================================================================
# Get similar words of keywords from text and print them with amount
#===============================================================================
from __future__ import print_function
import sys
from os import listdir
from os.path import exists, join

################################################################################
def distance(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
 
    current_row = range(n+1) # Keep current and previous row, not entire matrix
    for i in range(1, m+1):
        previous_row, current_row = current_row, [i]+[0]*n
        for j in range(1,n+1):
            add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]
            if a[j-1] != b[i-1]:
                change += 1
            current_row[j] = min(add, delete, change)
 
    return current_row[n]

################################################################################
def get_similar_words(text, keywords):
    """Get similar words of keywords from text."""
    similar_words = {}
    max_keyw_len = max(len(word) for word in keywords)
    
    for l in ".,!?:-='\"":     # delete punctuation from text
        text = text.replace(l, '')
    
    words = text.split()    # get list of words from text
    
    for word in words:
        len_word = len(word)
        # discard extra long words
        if len_word > 0 and len_word < max_keyw_len + 4:
            for keyw in keywords:
                if len(keyw) > 0 and abs(len_word - len(keyw)) < 4 \
                                    and distance(word, keyw) in range(2):
                    if similar_words.has_key(keyw): # keyw in dict
                        for i, (w, cnt) in enumerate(similar_words[keyw]):
                            if w == word: # if word in tuple then cnt++
                                similar_words[keyw][i] = (w, cnt + 1)
                                break
                        else: # if word not in list of tuples then append
                            similar_words[keyw].append((word, 1))
                    else: # add new list to dict if keyw not in dict
                        similar_words[keyw] = [(word, 1)]
                     
    return similar_words

################################################################################
def get_similar_words_count(text, keywords):
    """Get similar words of keywords from text."""
    count = 0
    max_keyw_len = max(len(word) for word in keywords)
    
    for l in ".,!?:-='\"":     # delete punctuation from text
        text = text.replace(l, '')
    
    words = text.split()    # get list of words from text
    
    for word in words:
        len_word = len(word)
        # discard extra long words
        if len_word > 0 and len_word < max_keyw_len + 4:
            isabnormal = False
            for keyw in keywords:
                if len(keyw) > 0 and abs(len_word - len(keyw)) < 4 \
                                    and distance(word, keyw) in range(2):
                    count += 1
                    isabnormal = True
                    
                if (isabnormal):
                    break
    return count


################################################################################
def print_similar_words(text_file, keywords_file="keywords.txt"):
    """Print keywords and similar words in text."""
    with open(keywords_file, "r") as f:
        keywords = f.read().split('\n')
    
    with open(text_file, "r") as f:
        text = f.read()
        similar_words = get_similar_words(text, keywords)
        for key in similar_words:
            print(key, end=': ')     # print keyword
            for pair in similar_words[key]:
                print(pair, end=' ') # print pairs of similar keyword with amount
            print()

################################################################################
# main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter email files folder name in command line:")
        print("sim_keywords_folder.py foldername\n")
        print("If you want set a path of keywords file (default: 'keywords.txt'):")
        print("sim_keywords_folder.py foldername keywords_another.txt")
        sys.exit(1)
        
    if not exists(sys.argv[1]):
        print("Folder %s is not exist" % sys.argv[1])
    else:
        for file in sorted(listdir(sys.argv[1])):
            if len(sys.argv) < 3:
                keywords_file = "keywords.txt"
            else:
                keywords_file = sys.argv[2]
            
            print("\nFile: %s" % file)
            print_similar_words(join(sys.argv[1], file), keywords_file)        
