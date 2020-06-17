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
import csv

def cwd():
    return os.getcwd().replace("\\", "/")

with open("dictionary_corpus.pickle", 'rb') as f:
    dictionary = pickle.load(f)
    corpus = pickle.load(f)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


###############################################################################
# Training
# --------
# 
# We are ready to train the LDA model. We will first discuss how to set some of
# the training parameters.
# 
# First of all, the elephant in the room: how many topics do I need? There is
# really no easy answer for this, it will depend on both your data and your
# application. I have used 10 topics here because I wanted to have a few topics
# that I could interpret and "label", and because that turned out to give me
# reasonably good results. You might not need to interpret all your topics, so
# you could use a large number of topics, for example 100.
# 
# ``chunksize`` controls how many documents are processed at a time in the
# training algorithm. Increasing chunksize will speed up training, at least as
# long as the chunk of documents easily fit into memory. I've set ``chunksize =
# 2000``, which is more than the amount of documents, so I process all the
# data in one go. Chunksize can however influence the quality of the model, as
# discussed in Hoffman and co-authors [2], but the difference was not
# substantial in this case.
# 
# ``passes`` controls how often we train the model on the entire corpus.
# Another word for passes might be "epochs". ``iterations`` is somewhat
# technical, but essentially it controls how often we repeat a particular loop
# over each document. It is important to set the number of "passes" and
# "iterations" high enough.
# 
# I suggest the following way to choose iterations and passes. First, enable
# logging (as described in many Gensim tutorials), and set ``eval_every = 1``
# in ``LdaModel``. When training the model look for a line in the log that
# looks something like this::
# 
#    2016-06-21 15:40:06,753 - gensim.models.ldamodel - DEBUG - 68/1566 documents converged within 400 iterations
# 
# If you set ``passes = 20`` you will see this line 20 times. Make sure that by
# the final passes, most of the documents have converged. So you want to choose
# both passes and iterations to be high enough for this to happen.
# 
# We set ``alpha = 'auto'`` and ``eta = 'auto'``. Again this is somewhat
# technical, but essentially we are automatically learning two parameters in
# the model that we usually would have to specify explicitly.
# 

start_time = time.time()

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 49
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

end_time = time.time()
print('Clustering took {:.2f} s'.format(end_time - start_time))

###############################################################################
# We can compute the topic coherence of each topic. Below we display the
# average topic coherence and print the topics in order of topic coherence.
# 
# Note that we use the "Umass" topic coherence measure here (see
# :py:func:`gensim.models.ldamodel.LdaModel.top_topics`), Gensim has recently
# obtained an implementation of the "AKSW" topic coherence measure (see
# accompanying blog post, http://rare-technologies.com/what-is-topic-coherence/).
# 
# If you are familiar with the subject of the articles in this dataset, you can
# see that the topics below make a lot of sense. However, they are not without
# flaws. We can see that there is substantial overlap between some topics,
# others are hard to interpret, and most of them have at least some terms that
# seem out of place. If you were able to do better, feel free to share your
# methods on the blog at http://rare-technologies.com/lda-training-tips/ !
# 

top_topics = model.top_topics(corpus, topn=num_topics) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
#pprint(top_topics)

#https://www.tutorialspoint.com/reading-and-writing-csv-file-using-python
topics_csv_file = open('topic_words.csv','w', newline='')
fw = csv.writer(topics_csv_file)

for topic in top_topics:
    topic_words = []
    for freq_word in topic[0]:
        topic_words.append(freq_word[1])
    fw.writerow(topic_words)

topics_csv_file.close()

###############################################################################
# Things to experiment with
# -------------------------
# 
# * ``no_above`` and ``no_below`` parameters in ``filter_extremes`` method.
# * Adding trigrams or even higher order n-grams.
# * Consider whether using a hold-out set or cross-validation is the way to go for you.
# * Try other datasets.
#
# Where to go from here
# ---------------------
#
# * Check out a RaRe blog post on the AKSW topic coherence measure (http://rare-technologies.com/what-is-topic-coherence/).
# * pyLDAvis (https://pyldavis.readthedocs.io/en/latest/index.html).
# * Read some more Gensim tutorials (https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials).
# * If you haven't already, read [1] and [2] (see references).
# 
# References
# ----------
#
# 1. "Latent Dirichlet Allocation", Blei et al. 2003.
# 2. "Online Learning for Latent Dirichlet Allocation", Hoffman et al. 2010.
# 

#for doc_bow in corpus:
#    pprint(model.get_document_topics(doc_bow))

with open("topics_by_document.pickle", 'wb') as f:
    pickle.dump(model, f)
    pickle.dump(corpus, f)
    