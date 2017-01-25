# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:42:29 2016

@author: abzooba
"""

# THE imports
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import codecs
#from conf import config
#from collections import OrderedDict
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from gensim.models import Phrases
from gensim.models.word2vec import Word2Vec
import re

# Memory friendly iterator that uses wiki API to retrieve content from a wiki page
class WikiMedicalIter(object):
    def __init__(self, directory, sentence_tokenize = False):
        self.directory = directory
        self.sentence_tokenize = sentence_tokenize
    
    def __iter__(self):
        print 'pass...'
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                file_name = os.path.join(root, f )
                file_present = os.path.isfile(file_name)
                filename, file_extension = os.path.splitext(f)

                if file_extension == '.txt' and file_present:
                    # print filename
                    file_wiki = codecs.open(file_name, "r", "utf-8")
                    content = file_wiki.read()
                    file_wiki.close()

                    if self.sentence_tokenize:
                        sent_tokenize_list = sent_tokenize(content)
                        for sent in sent_tokenize_list:
                            words = self.sentenceCleaner(sent)
                            if len(words) > 2:
                                yield words
                    else:
                        yield content.lower().split()
    
    def sentenceCleaner( self, sentence, remove_stopwords=False, clean_special_chars_numbers = True, \
                        lemmatize = False, stem = False, stops = set(stopwords.words("english")) ):
        """
        Function to convert a document to a sequence of words, optionally removing stop words.  Returns a list of words.
    
        :param sentence:
        :param remove_stopwords:
        :param clean_special_chars_numbers:
        :param lemmatize:
        :param stem:
        :param stops:
        :return:
        """
        words = []
        
        if sentence.startswith('==') == False:
            sentence_text = sentence
            
            # Optionally remove non-letters (true by default)
            if clean_special_chars_numbers:
                sentence_text = re.sub("[^a-zA-Z]"," ", sentence_text)
            
            # Convert words to lower case and split them
            words = sentence_text.lower().split()
            
            # Optional stemmer
            if stem:
                stemmer = PorterStemmer()
                words = [ stemmer.stem(w) for w in words ]
        
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                words = [ lemmatizer.lemmatize(w) for w in words ]
            
            # Optionally remove stop words (false by default)
            if remove_stopwords:
                words = [w for w in words if not w in stops]
        
        # 4. Return a list of words
        return(words)

    def fileCounter(self):
        file_counter = 0
        for root, dirs, files in os.walk(self.directory):
            file_counter += len(files)
        return file_counter

def testIterator():
    scraper = WikiMedicalIter('../data', True) # a memory-friendly iterator
    for content in scraper:
        print content

def showUsage():
    print 'Usage:'
    print 'python <filename> <dirname> <modelname>'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        showUsage()
        exit()

    folder_name = sys.argv[1] #'../data/List of disorders'
    iterator = WikiMedicalIter(folder_name, True) # a memory-friendly iterator
    noOfArticles = iterator.fileCounter()
    print 'Given folder has', noOfArticles, 'articles.'
    
    bigram_transformer = Phrases(iterator)
    # trigram_transformer = Phrases(bigram_transformer[iterator])

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # model = Word2Vec(trigram_transformer[bigram_transformer[iterator]], workers=num_workers, \
    #             size=num_features, min_count = min_word_count, \
    #             window = context, sample = downsampling, seed = 786)
    model = Word2Vec(bigram_transformer[iterator], workers=num_workers, \
               size=num_features, min_count = min_word_count, \
               window = context, sample = downsampling, seed = 786)
    # model = Word2Vec(iterator, workers=num_workers, \
    #            size=num_features, min_count = min_word_count, \
    #            window = context, sample = downsampling, seed = 786)

    print model.syn0.shape

    # to trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model_name = sys.argv[2] # + str( noOfArticles ) + '_bi'
    model.save(model_name)
    #model.save_word2vec_format(model_name)