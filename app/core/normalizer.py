from __future__ import division

from app.conf import appconfig
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import extractChanges
from nltk.util import ngrams
import logging

class Normalizer(object):
    def __init__(self):
        self.words_list = []
        self.remove_stopwords = appconfig.REMOVE_STOPWORDS
        self.remove_nos = appconfig.REMOVE_NUMBERS

    def sentenceCleaner(self, sentence, remove_stopwords=False, clean_special_chars_numbers = True, \
                        lemmatize = False, stem = False, stops = set(stopwords.words("english"))):
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
        words_extended = []
        sentence=unicode(sentence,encoding='iso-8859-1')
        if sentence.startswith('==') == False:
            sentence_text = sentence

            # Convert words to lower case and split them
            words = sentence_text.lower().split()
            sentence_text = ' '.join(words)

            # Optionally remove non-letters (true by default)
            if clean_special_chars_numbers:
                sentence_text = re.sub("[^a-zA-Z_]", " ", sentence_text)
                words = sentence_text.split()
                # print words

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

            # Also splitting on the basis of '/'
            for word in words:
                if '/' in word and not bool(re.search(r'\d', word)):
                    words_extended.extend([w for w in word.split('/') if w is not ''])
                    continue
                else:
                    words_extended.append(word)

        # 4. Return a list of words
        return(words_extended)

    def normalize(self, df, column, original, destination_file):
        self.words_list = []

        # add check if column exists
        if column in df.columns:
            column_data = np.array(df[column].values)

            normalized_column = []
            for claim in column_data:
                cleaned_claim = self.sentenceCleaner(claim, self.remove_stopwords, self.remove_nos)
                self.words_list.extend(cleaned_claim)
                normalized_column.append(' '.join(cleaned_claim))

            df.loc[:, column] = normalized_column
        else:
            df.loc[:, column] = df[original].str.lower()


        s = pd.Series(self.words_list)
        s = s.value_counts()
        # s.to_csv(destination_file)

        return df

    def word_grams(self, sentence, mini=2, maxi=6):
        sentence = sentence.split()
        s = []
        for n in range(mini, maxi+1):
            for ngram in ngrams(sentence, n):
                s.append(' '.join(str(i) for i in ngram))
        return s

    def addUnderscoreAbbreviationForMRdata(self,df,column):
        dbconnect = extractChanges.DBconnect()
        expansions = dbconnect.getAbbreviationsFromTable()['Expansion'].values
        dbconnect.closeConnection()
        corrected_norm_desc_new = []
        # don't use pandas, use numpy to optimize...
        # for index, row in df.iterrows():
        #     corrected_norm_desc = row[column]
        #     grams = self.word_grams(corrected_norm_desc)
        #     for ngram in grams:
        #         if any(expansions==ngram):
        #             corrected_norm_desc = corrected_norm_desc.replace(ngram,ngram.replace(' ','_'))
        #     corrected_norm_desc_new.append(corrected_norm_desc)
        # df[column] = corrected_norm_desc_new

        # Used numpy instead of pandas
        corrected_norm_descriptions = df[column].values
        for index, corrected_norm_desc in enumerate(corrected_norm_descriptions):
            # corrected_norm_desc = row[column]
            if corrected_norm_desc is not None:
                grams = self.word_grams(corrected_norm_desc)
                for ngram in grams:
                    if any(expansions==ngram):
                        corrected_norm_desc = corrected_norm_desc.replace(ngram,ngram.replace(' ','_'))
                corrected_norm_desc_new.append(corrected_norm_desc)
        df[column] = corrected_norm_desc_new
        return df