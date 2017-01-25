from app.conf import appconfig
from app.core import learner
import pandas as pd
from app.core import feature_mapper
import pickle
import collections
import numpy as np
from scipy.sparse import hstack
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import os
from datetime import datetime
import operator
import distutils
import tempfile
import logging
import shutil

class ClassificationEngine(object):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.learner = learner.Learner()
        self.featureExtractionFn = appconfig.featureExtractionFunctions
        self.featuresFile = appconfig.combinedfiles
        # self.plots_folder = appconfig.PLOTS_FOLDER
        self.archived_folder = appconfig.MODELS_FOLDER_ARCHIVED
        self.datetime_format = appconfig.DATETIME_FORMAT
        self.models_folder_learning = appconfig.MODELS_FOLDER_LEARNING
        self.models_folder_classification = appconfig.MODELS_FOLDER_CLASSIFICATION
        self.number_of_models = appconfig.NUMBER_OF_MODELS
        self.classifiersFile=appconfig.CLASSIFIER_FILE
        self.decideModelsFolder()
        if classifiers:
            if os.path.exists('../' + self.models_folder_classification+'/'+self.classifiersFile+'.pickle'):
                classifiers_final = collections.OrderedDict()
                f = open('../' + self.models_folder_classification+'/'+self.classifiersFile+'.pickle', 'rb')
                classifiersFinal = pickle.load(f)
                f.close()
                for clf_name in classifiersFinal:
                    classifiers_final[clf_name]=classifiers[clf_name]
            else:
                classifiers_final=classifiers
            self.classifiers=classifiers_final

    def decideModelsFolder(self):

        logging.info(' : '+__file__+' : : In decideModelFolder()')
        # Check if training models folder has all pickle files
        learning_model_list = [s for s in os.listdir('../' + self.models_folder_learning) if s.endswith('.pickle')]
        logging.debug(' : '+__file__+' : : number_of_models = ' + str(self.number_of_models))
        logging.debug(' : '+__file__+' : : learning_model_list = ' + str(len(learning_model_list)))
        if len(learning_model_list) == self.number_of_models:
            logging.debug(' : '+__file__+' : : picking classification models from models_folder_learning')
            distutils.dir_util.copy_tree('../' + self.models_folder_learning, '../'+self.models_folder_classification)
        else:
            logging.debug(' : '+__file__+' : : picking models from archived_folder')
            dir_list = os.listdir('../'+self.archived_folder+'/')
            dates_list = [datetime.strptime(date, self.datetime_format) for date in dir_list]
            index, value = max(enumerate(dates_list), key=operator.itemgetter(1))
            latest_archive = datetime.strftime(value,self.datetime_format)
            distutils.dir_util.copy_tree('../' + self.archived_folder + '/'+latest_archive, '../'+self.models_folder_classification)
            logging.info(' : '+__file__+' : : Classification models picked from archived_models folder:'+ str(latest_archive)+' \n')
        self.dest_folder = self.models_folder_classification

    def loadSelectedFeatures(self, selected_features_file):
        return self.loadClassifier('../' + self.dest_folder + '/' + selected_features_file)

    def loadImportantWords(self, important_words_file):
        imp_words = []
        try:
            imp_words = self.loadClassifier('../' + self.dest_folder + '/' + important_words_file)
        except Exception as e:
            imp_words = []

        return imp_words

    def loadClassifier(self, model_name):
        f = open(model_name + '.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()

        return classifier

    def loadMapper(self, mapper_name):
        self.mapper = self.loadClassifier('../' + self.dest_folder + '/' + mapper_name)

    def loadClassifiers(self):
        for clf_name in self.classifiers:
            classifier = self.loadClassifier('../' + self.dest_folder + '/' + clf_name)
            self.classifiers[clf_name] = classifier

    def loadEncoder(self, encoder):
        self.label_encoder = self.loadClassifier('../' + self.dest_folder + '/' + encoder)

    def loadEnsembler(self, ensemble_file):
        self.ensembler = self.loadClassifier('../' + self.dest_folder + '/' + ensemble_file)

    def loadWord2vecModel(self, fname, f):
        self.w2vModel = Word2Vec.load("../" + fname +"/" + f)

    def featureMapper(self, df,normdesc,selected_features,w2vfile):

        mapper=feature_mapper.FeatureMapper( self.mapper)
        df = getattr(mapper, self.featureExtractionFn)(df, self.featuresFile, w2vfile)
        return df[selected_features]

    def classify(self, df, original_feature, normdesc, selected_features, w2vfile):
        logging.info(' : '+__file__+' : : classify()')
        df_vectors = self.featureMapper(df,normdesc,selected_features,w2vfile)
        vectors = df_vectors.values
        logging.debug(' : '+__file__+' : : reduced vectors shape = ' + str(vectors.shape))
        descs = df[original_feature].values

        for clf_name in self.classifiers:
            logging.info(' : '+__file__+' : : classifying : ' + clf_name)
            classifier = self.classifiers[clf_name]

            if clf_name in appconfig.syntactic_classifiers:
                predictions, probabilities = self.learner.classify(clf_name, classifier, descs)
            else:
                predictions, probabilities = self.learner.classify(clf_name, classifier, vectors)

            if clf_name == 'XGBoost':
                predictions = self.label_encoder.inverse_transform(predictions)

            df['Predicted_Class_'+clf_name] = predictions
            df['Prob_'+clf_name] = probabilities

            # TODO boosting confidences...for Logistic and Levenshtein
            if clf_name == 'Jaro':
                df.loc[(df['Prob_'+clf_name] >= 0.8) & (df['Prob_'+clf_name] < 0.85), 'Prob_'+clf_name] = 0.85
            elif clf_name == 'Logistic_OvR':
                df.loc[(df['Prob_'+clf_name] >= 0.8) & (df['Prob_'+clf_name] < 0.9), 'Prob_'+clf_name] = 0.96
                df.loc[(df['Prob_'+clf_name] >= 0.7) & (df['Prob_'+clf_name] < 0.8), 'Prob_'+clf_name] = 0.92
                df.loc[(df['Prob_'+clf_name] >= 0.6) & (df['Prob_'+clf_name] < 0.7), 'Prob_'+clf_name] = 0.76
            elif clf_name == 'DLevenshtein':
                df.loc[(df['Prob_'+clf_name] >= 0.8) & (df['Prob_'+clf_name] < 0.9), 'Prob_'+clf_name] = 0.96
                df.loc[(df['Prob_'+clf_name] >= 0.7) & (df['Prob_'+clf_name] < 0.8), 'Prob_'+clf_name] = 0.93
                df.loc[(df['Prob_'+clf_name] >= 0.6) & (df['Prob_'+clf_name] < 0.7), 'Prob_'+clf_name] = 0.88
                df.loc[(df['Prob_'+clf_name] >= 0.5) & (df['Prob_'+clf_name] < 0.6), 'Prob_'+clf_name] = 0.81

        return df

    def votePredictions(self, df):
        logging.info(' : '+__file__+' : : Voting predictions...')

        # get Voted Output
        voter = learner.Voter(df, self.classifiers)
        df = voter.votingPredictions()

        return df

    def ensemblePredictions(self, df):
        logging.info(' : '+__file__+' : : Ensemble predictions...')
        for clf_name in self.classifiers:
            col_name = 'Predicted_Class_'+clf_name
            df[col_name] = self.label_encoder.transform(df['Predicted_Class_'+clf_name].values)

        ensemble_x = df[[k for k in df.columns if 'Predicted_' in k]].values

        voted_output = self.ensembler.predict(ensemble_x)
        df['Voted Output'] = voted_output
        df['Sum Correct'] = 0
        for clf_name in self.classifiers:
            col_name = 'Predicted_Class_'+clf_name
            df[col_name] = self.label_encoder.inverse_transform(df[col_name].values)

            df['Correct_'+clf_name] = 0
            df.loc[ df[col_name] == df['Voted Output'], 'Correct_'+clf_name ] = 1
            df['Sum Correct'] += df['Correct_'+clf_name]

        print 'Wrongly ensembled cases:', len(df.loc[df['Sum Correct'] == 0])
        return df

