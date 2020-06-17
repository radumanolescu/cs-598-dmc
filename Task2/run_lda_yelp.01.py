# -*- coding: utf-8 -*-
r"""
Created on Wed Feb 12 19:53:32 2020

@author: Radu

LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the Yelp corpus.

"""

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
# https://www.geeksforgeeks.org/python-map-function/
# https://www.geeksforgeeks.org/python-string-replace/
# https://www.w3schools.com/python/python_file_open.asp
# https://stackoverflow.com/questions/491921/unicode-utf-8-reading-and-writing-to-files-in-python

import time
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import io
import os.path
import re
import tarfile

import smart_open
import os
import pickle

def cwd():
    return os.getcwd().replace("\\", "/")

def extract_documents(rootDir):
    arr = map(lambda x: rootDir + x, os.listdir("rvwByCsne"))
    files = list(arr)
    #print(files)
    for fn in files:
        fh = open(fn, "r", encoding="utf-8")
        #rint(fn + " : " + fh.read(32))
        print("---> " + fn)
        yield fh.read()
    fh.close()

time47 = time.time()
docs = list(extract_documents(cwd()+"/rvwByCsne/"))

print(len(docs))
print(docs[0][:500])    

# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

###############################################################################
# We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
# stemmer in this case because it produces more readable words. Output that is
# easy to read is very desirable in topic modelling.
# 

# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

###############################################################################
# We find bigrams in the documents. Bigrams are sets of two adjacent words.
# Using bigrams we can get phrases like "machine_learning" in our output
# (spaces are replaced with underscores); without bigrams we would only get
# "machine" and "learning".
# 
# Note that in the code below, we find bigrams and then add them to the
# original data, because we would like to keep the words "machine" and
# "learning" as well as the bigram "machine_learning".
# 
# .. Important::
#     Computing n-grams of large dataset can be very computationally
#     and memory intensive.
# 


# Compute bigrams.
from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

###############################################################################
# We remove rare words and common words based on their *document frequency*.
# Below we remove words that appear in less than 20 documents or in more than
# 50% of the documents. Consider trying to remove words only based on their
# frequency, or maybe combining that with this approach.
# 

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

###############################################################################
# Finally, we transform the documents to a vectorized form. We simply compute
# the frequency of each word, including the bigrams.
# 

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

###############################################################################
# Let's see how many tokens and documents we have to train on.
# 

time135 = time.time()
print('Dictionary & corpus prep took {:.2f} s'.format(time135 - time47))

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

with open("dictionary_corpus.pickle", 'wb') as f:
    pickle.dump(dictionary, f)
    pickle.dump(corpus, f)

