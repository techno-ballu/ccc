import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.conf import appconfig
from app.core import data_preprocessor
from app.core import normalizer
from app.core import classifier
from app.core import learner
from datetime import datetime
from app.core import updateTables
from app.core import feature_mapper
import numpy as np
from sklearn.metrics import accuracy_score
import logging
import time
import glob
start = time.time()
import  pickle
import collections

print (time.strftime("%H:%M:%S"))

# suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

target = appconfig.TARGET
data_folder = appconfig.MODELS_FOLDER_CLASSIFICATION
unlabeled_file = appconfig.UNLABELED_FILE
vectorizer_file = appconfig.VECTORIZER_FILE
ensembler_file = appconfig.ENSEMBLER_FILE
encoder_file = appconfig.ENCODER_FILE
important_words_file = appconfig.IMPORTANT_WORDS_FILE
selected_features_file=appconfig.SELECTED_FEATURES_FILE
text_feature = appconfig.TEXT_FEATURE
original_feature = appconfig.ORIGINAL_DESCRIPTION
remove_stopwords = appconfig.REMOVE_STOPWORDS
remove_numbers = appconfig.REMOVE_NUMBERS
# to_be_removed = appconfig.TO_BE_REMOVED
classifiers = appconfig.classifiers
w2v_unlabeledfile = appconfig.W2V_UNLABELED_FILE
vocab_trainfile = appconfig.VOCABULARY_TRAIN_FILE
vocab_testfile = appconfig.VOCABULARY_TEST_FILE


def classifyUnlabeled():


    startClassification=time.time()
    # combination='B'
    # print appconfig.MODELS_FOLDER_LEARNING
    dest_folder = appconfig.MODELS_FOLDER_CLASSIFICATION
    classifiersFile=appconfig.CLASSIFIER_FILE
    featureExtractionFn = appconfig.featureExtractionFunctions
    featureFn = appconfig.featureFunctions
    classifiers_final=classifiers




    preprocessor = data_preprocessor.DataPreprocessor(target, data_folder, dest_folder, None, unlabeled_file) #
    logging.info(' : '+__file__+' : '+'Preprocessor instantiated')
    preprocessor.loadData()
    data = preprocessor.saveData()


    df_unlabelled_copy = data['testing']

    # backup normalized description
    normalized_desc_unlabelled = pd.DataFrame(df_unlabelled_copy[text_feature])

    # clean claim lines using sentence cleaner
    startNormalize=time.time()
    data_normalizer = normalizer.Normalizer()
    df_unlabelled = data_normalizer.normalize(df_unlabelled_copy, text_feature, original_feature, '../' + dest_folder + '/' + vocab_trainfile)
    print "time taken in normalization inside python:"
    print time.time()-startNormalize

    # instantiate the classification engine and load models, vectorizer, encoder & ensembler.
    startLoadModels=time.time()
    classification_engine = classifier.ClassificationEngine(classifiers_final)
    if os.path.exists('../' + dest_folder+'/'+classifiersFile+'.pickle'):
        classifiers_final = collections.OrderedDict()
        f = open('../' + dest_folder+'/'+classifiersFile+'.pickle', 'rb')
        classifiersFinal = pickle.load(f)
        f.close()
        for clf_name in classifiersFinal:
            logging.info(' : '+__file__+' : loading...' + str(clf_name))
            classifiers_final[clf_name]=classifiers[clf_name]
    else:
        classifiers_final=classifiers
    logging.info(' : '+__file__+' : '+'ClassificationEngine instantiated')
    classification_engine.loadClassifiers()
    # classification_engine.loadEnsembler(ensembler_file)
    classification_engine.loadMapper(vectorizer_file)
    classification_engine.loadEncoder(encoder_file)
    imp_words = classification_engine.loadImportantWords(important_words_file)
    selected_features = classification_engine.loadSelectedFeatures(selected_features_file)
    print "Time taken in loading models and other binaries:"
    print time.time()-startLoadModels


    # classify claim lines
    startClassify=time.time()
    logging.info(' : '+__file__+' : '+'Classification started')
    df_unlabelled = classification_engine.classify(df_unlabelled, original_feature, text_feature, selected_features, w2v_unlabeledfile)
    logging.info(' : '+__file__+' : '+'Classification done.')
    print "Time taken in classification:"
    print time.time()-startClassify

    # ensemble predictions from models to get voted output
    startEnsemble=time.time()
    df_unlabelled = classification_engine.votePredictions(df_unlabelled)
    logging.info(' : '+__file__+' : '+'Voting done')
    print "Time taken in voting:"
    print time.time()-startEnsemble

    startCalculation=time.time()
    confidence = learner.ConfidenceCalculation(classifiers_final, df_unlabelled)
    df_unlabelled = confidence.calculateByBoosting()
    print "TIme taken in confidence Calculation:"
    print time.time()-startCalculation

    startFeatureExtraction=time.time()
    feature_impor = learner.FeatureImportance(imp_words)
    logging.info(' : '+__file__+' : '+'Feature extraction started')
    df_unlabelled = feature_impor.extractFeatures(df_unlabelled, text_feature)
    logging.info(' : '+__file__+' : '+'Feature extraction done')
    df_unlabelled[text_feature] = normalized_desc_unlabelled

    # df_unlabelled.to_csv('../' + dest_folder + '/Predictions_' + 'BW' + '.csv', index=False)
    print "Time taken in feature extraction:"
    print time.time()-startFeatureExtraction

    df_unlabelled.to_csv('../' + dest_folder + '/df_unlabelled.csv', index=False)

    # TODO uncomment files deletion
    # for csv_file_path in glob.glob('../' + data_folder + "/*.csv"):
    #     os.remove(csv_file_path)

    df_result=df_unlabelled
    classification_date = str(datetime.now())
    df_result['Classification Date']=classification_date
    # result=df_result[appconfig.CLASSIFICATION_COLUMN_TO_UPLOAD]

    print '-'*100, '\n'
    print "total time taken in running classification pipeline:"
    print time.time()-startClassification
    return classification_date, df_result

if __name__ == '__main__':
    classifyUnlabeled()
    print (time.strftime("%H:%M:%S"))
    done = time.time()
    elapsed = done - start
    print 'elapsed = ', elapsed