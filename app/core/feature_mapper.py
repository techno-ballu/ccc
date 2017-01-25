from app.conf import appconfig
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import re
import os.path
# from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import itertools
import time
from scipy.stats.stats import pearsonr
import logging

class FeatureMapper(object):
    def __init__(self, transformer = None, dimTransformer=None):
        self.bow_max_feats = 1000
        self.ngram_range = appconfig.BOW_N_GRAM_RANGE
        self.chargeClassFeatures = appconfig.medical_concepts.keys()
        self.chargeClasses = appconfig.medical_concepts
        self.text_feature = appconfig.TEXT_FEATURE
        self.original_feature = appconfig.ORIGINAL_DESCRIPTION
        self.target = appconfig.TARGET
        self.dest_folder = appconfig.MODELS_FOLDER_LEARNING
        self.data_folder = appconfig.DATA_MODELS_LEARNING
        self.word2vec_folder = appconfig.WORD2VEC_FOLDER
        self.wikimodelname = appconfig.WIKIMODEL
        self.count_vectorizer = CountVectorizer(analyzer = "word",           \
                                     preprocessor = None,       \
                                     stop_words = None,    \
                                     binary = False,    \
                                     # decode_error="ignore" ,          \
                                     # max_features = self.bow_max_feats,      \
                                     ngram_range = self.ngram_range)

        self.transformer = transformer
        self.std_dev_cutoff = appconfig.STD_DEV_CUTOFF
        self.coll_cutoff = appconfig.CORRELATION_CUTOFF
        self.wikimodel = self.loadWord2vecModel(self.word2vec_folder,self.wikimodelname)
        # self.wikivocab = np.array([i[0] for i in self.wikimodel.vocab.items()])
        # print 'length of wiki vocab = ', self.wikivocab.size

    def trainVectorizer(self, sentences):
        logging.info(' : '+__file__+' : '+'In trainVectorizer() method.')
        self.transformer = self.count_vectorizer.fit(sentences)
        print 'stop words:'
        print list(self.transformer.stop_words_)

        self.bow_max_feats=len(self.transformer.vocabulary_.keys())
        logging.debug(' : '+__file__+' : '+'bow max length : '+str(self.bow_max_feats))
        print len(self.transformer.vocabulary_.keys()), 'long vocabulary...'
        logging.debug(' : '+__file__+' : '+'vocabulary length: '+str(len(self.transformer.vocabulary_.keys())))

    def transform(self, sentences):
        # sentences=str(sentences)
        transformed_matrix = self.transformer.transform(sentences)
        return transformed_matrix

    def bowFeatures(self):
        return self.transformer.get_feature_names()

    def w2vFeatures(self):
        return self.chargeClassFeatures

    def saveFeaturesFile(self, vectors, features, df, combined_file):

        # prepare combined training set
        df_combined = pd.DataFrame(vectors, columns=features, index=df.index)

        if self.original_feature in df:
            df_combined[self.original_feature] = df[self.original_feature]

        if self.text_feature in df:
            df_combined[self.text_feature] = df[self.text_feature]

        if self.target in df:
            df_combined[self.target] = df[self.target]

        # df_combined.to_csv('../' + self.dest_folder + '/' + combined_file, index=False)
        return df_combined

    def saveFeatureMapper(self, mapper_name):
        # Pickle BOW transformer
        f = open('../' + self.dest_folder + '/' + mapper_name + '.pickle', 'wb')
        pickle.dump(self.transformer, f)
        f.close()

    def saveSelectedFeatures(self, features_list_file, selected_features):
        f = open('../' + self.dest_folder + '/' + features_list_file + '.pickle', 'wb')
        pickle.dump(selected_features, f)
        f.close()

    def loadWord2vecModel(self, fname, f):
        model = Word2Vec.load("../" + fname +"/" + f)
        return model

    def extractWord2VecFeatures(self, df, combined_file, w2vfile):
        vectors = self.freshExtractWord2VecFeatures(df, combined_file,w2vfile)
        vectors_meta = vectors[self.w2vFeatures()].values

        return vectors_meta

    def freshExtractWord2VecFeatures(self, df, combined_file, w2vfile):
        logging.info(' : '+__file__+' : '+'In freshExtractWord2VecFeatures() method.')
        if self.target in df.columns:
            data = df[[self.text_feature, self.original_feature, self.target]]
        else:
            data = df[[self.text_feature, self.original_feature]]

        # for charge_class in self.chargeClasses:
        #     data.loc[:,charge_class]=0.0

        text_column = np.array( data[self.text_feature].values )

        stops = set(stopwords.words("english"))
        distances_list = []
        # for index, row in data.iterrows():
        for i, row_desc in enumerate(text_column):
            distances = []
            word_list=[]
            # row_desc = row[self.text_feature]
            words = re.sub("[^a-zA-Z]"," ", row_desc.lower())
            words = [i for i in words.split() if i not in stops and len(i)>1]
            # word_list = [[w] for w in words if w in self.wikivocab]
            for word in words:
                try:
                    word=[word]
                    self.wikimodel.n_similarity(word,self.chargeClasses['Painkillers'])
                    word_list.append(word)
                except Exception:
                    continue

            # print word_list
            for charge_class in self.chargeClasses:
                similarity = 0.0
                formatted = ['_'.join(value.lower().split()) for value in self.chargeClasses[charge_class]]

                if word_list != []:
                    similarity=self.wikimodel.n_similarity(word_list, formatted)

                if isinstance(similarity, float):
                    distances.append(similarity)
                else:
                    distances.append(similarity[0])

            maxi=max(distances)
            mini=min(distances)
            for x, distance in enumerate(distances):
                if (maxi-mini) > 0:
                    distances[x] = (distance - mini) / (maxi - mini)

            distances_list.append(distances)

        distances_list = np.array(distances_list)

        for i, charge_class in enumerate(self.chargeClasses):
            data.loc[:,charge_class] = distances_list[:,i]

        # data.to_csv("../"+self.data_folder+"/"+w2vfile.split('.')[0] + '_debug.csv', index=False)
        data=data[self.chargeClasses.keys()]
        # data.to_csv("../"+self.data_folder+"/"+w2vfile, index=False)

        return data

    def extractBagOfWordsFeatures(self, df, combined_file,w2vfile):
        # Feature names
        features_bow = self.bowFeatures()
        vectors_bow = self.transformer.transform(df[self.text_feature].values).toarray()

        return vectors_bow

    def extractMetamapFeatures(self, df,combined_file,w2vfile):
        # Feature names
        features_meta = self.metafeatures
        vectors_meta = df[features_meta].values

        return vectors_meta

    def combineW2VBOW(self, df, combined_file = None,w2vfile = None):
        features_bow = self.bowFeatures()
        # print df[self.text_feature].values
        vectors_bow = self.transformer.transform(df[self.text_feature].values).toarray()

        features_w2v = self.w2vFeatures()
        vectors_w2v = self.extractWord2VecFeatures(df,combined_file,w2vfile)

        stacked = np.hstack([vectors_bow, vectors_w2v])

        return self.saveFeaturesFile(stacked, features_bow + features_w2v, df, combined_file)

        # return stacked

    def combinedFeaturesW2VBOW(self):
        return self.bowFeatures() + self.w2vFeatures()

    def removeOnlyZeroVariance(self, df_train_vectors, df_test_vectors, features = None, combined_file = None):
        logging.info(' : '+__file__+' : '+'In removeOnlyZeroVariance() method.')
        startZero=time.time()

        # calculate the standard deviation in training data
        std_dev = df_train_vectors.std()
        no_original_features = len(std_dev)
        std_dev = std_dev.loc[std_dev > self.std_dev_cutoff]
        selected_features = std_dev.index.values
        no_of_dropped_zero = no_original_features - len(selected_features)
        logging.debug(' : '+__file__+' : '+'# Dropped by Zero Variance = ' + str(no_of_dropped_zero))

        if len(selected_features) > 0:
            # select variant features
            df_train_vectors = df_train_vectors[selected_features]
            df_test_vectors = df_test_vectors[selected_features]

        logging.debug(' : '+__file__+' : Time taken in removing zero variance: ' + str(time.time()-startZero))
        print "Time taken in removing zero variance:"
        print time.time()-startZero

        return df_train_vectors.values, df_test_vectors.values, df_train_vectors.columns.tolist()

    def removeZeroVariance(self, df_train_vectors, df_test_vectors, features, combined_file):
        logging.info(' : '+__file__+' : '+'In removeZeroVariance() method.')
        startZero=time.time()

        # calculate the standard deviation in training data
        std_dev = df_train_vectors.std()
        no_original_features = len(std_dev)
        std_dev = std_dev.loc[std_dev > self.std_dev_cutoff]
        selected_features = std_dev.index.values
        no_of_dropped_zero = no_original_features - len(selected_features)
        logging.debug(' : '+__file__+' : '+'# Dropped by Zero Variance = ' + str(no_of_dropped_zero))

        if len(selected_features) > 0:
            # select variant features
            df_train_vectors = df_train_vectors[selected_features]
            df_test_vectors = df_test_vectors[selected_features]
        else:
            selected_features = df_train_vectors.columns

        logging.debug(' : '+__file__+' : Time taken in removing zero variance: ' + str(time.time()-startZero))
        print "Time taken in removing zero variance:"
        print time.time()-startZero
        startPWC=time.time()

        correlations = {}
        columns = selected_features.tolist()

        for col_a, col_b in itertools.combinations(columns, 2):
            colA = np.array(df_train_vectors.loc[:, col_a].values)
            colB = np.array(df_train_vectors.loc[:, col_b].values)
            correlations[col_a + '__' + col_b] = pearsonr(colA, colB)

        result = pd.DataFrame.from_dict(correlations, orient='index')
        result.columns = ['PCC', 'p-value']

        high_correlations = result.loc[result['PCC'].abs() >= self.coll_cutoff].index.values.tolist()
        high_correlations = set([i.split('__')[0] for i in high_correlations])

        result = result.loc[result['PCC'].abs() < self.coll_cutoff]
        logging.debug(' : '+__file__+' : '+'# Dropped by Pair Wise Correlation = ' + str(len(high_correlations)))
        df_train_vectors.drop(high_correlations, axis=1, inplace=True)
        df_test_vectors.drop(high_correlations, axis=1, inplace=True)

        logging.debug(' : '+__file__+' : Time taken in removing PWC: ' + str(time.time()-startPWC))
        result.to_csv('../' + self.dest_folder + '/' + 'correlations.csv', index=True)
        print "Time taken in removing PWC:"
        print time.time()-startPWC
        return df_train_vectors.values, df_test_vectors.values, df_train_vectors.columns.tolist()