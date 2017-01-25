import collections
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sknn.mlp import Classifier, Layer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from app.core.jaro_wrinkler_algo import JaroWrinklerClassifier
from app.core.damerau_levenshtein_algo import DamerauLevenshteinClassifier
from sklearn.grid_search import GridSearchCV
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
import pandas as pd
import numpy as np
import random
import logging
import os
import pickle
import sys
from datetime import datetime
import operator
__author__ = 'Abzooba'
random.seed(786)
np.random.seed(786)
JARO_WINKLER_THRESHOLD=0.85

NO_TRAINING_LINES_FILE = open('../../config/NumberTrainingLines.txt', 'r') # No. of lines to keep in training sample
NO_TRAINING_LINES=NO_TRAINING_LINES_FILE.read()
NO_TRAINING_LINES=NO_TRAINING_LINES.strip()
NO_TRAINING_LINES_FILE.close()
# NO_TRAINING_LINES=15000 # No. of lines to keep in training sample
ID = 'Id'
TARGET = 'Charge Class'
TEXT_FEATURE = 'Normalized Description'
ORIGINAL_DESCRIPTION = 'Original Description'
MAX_ARCHIVE_FOLDERS = 10
MIN_SAMPLE_SIZE = 10
MIN_TEST_TRAIN_SPLIT = 5
MIN_TRAINING_SIZE = 999
SPLIT_STEP_SIZE = 1
MAX_STEP_SIZE = 20
CHARGE_CLASS_BUFFER = 500
TRAINING_COLUMNS_TO_UPLOAD=[ORIGINAL_DESCRIPTION,TEXT_FEATURE,TARGET,'Flag','Feature 1', 'Feature 2', 'Feature 3','Feature 4']
TRAINING_COLUMNS_TO_FETCH=['Desciption Id',ORIGINAL_DESCRIPTION,TEXT_FEATURE,TARGET,'Flag','Create Date','Feature 1', 'Feature 2', 'Feature 3','Feature 4']
CLASSIFICATION_COLUMN_TO_UPLOAD=['Classification Date',ORIGINAL_DESCRIPTION,TEXT_FEATURE,'Voted Output','Confidence Score','Feature 1', 'Feature 2', 'Feature 3','Feature 4','Changes']
DATA_MODELS_LEARNING= 'learning_models' #
MODELS_FOLDER_LEARNING = DATA_MODELS_LEARNING + '/models'
MODELS_FOLDER_CLASSIFICATION= 'classification_models'
MODELS_FOLDER_ARCHIVED= 'archived_models'
WORD2VEC_FOLDER='word2vec'
DATETIME_FORMAT = '%m%d%Y%H%M%S'
LOG_LEVEL=logging.INFO
LEARNING_LOG_FILENAME='../../learning.log'
CLASSIFICATION_LOG_FILENAME='../../classification.log'
THRESHOLD_CONFIG_LOG='../../threshold.log'
AUTO_CLASSIFY_SIMILARS=True
WIKIMODEL='wikimodel_v3_tri_10'
# PREDICTIONS_FOLDER = 'predictions'
# PLOTS_FOLDER = DATA_MODELS_LEARNING + '/plots'
CONF_SCORE='Confidence Score'
SELECTED_FEATURES_FILE = 'selected_features'
IMPORTANT_WORDS_FILE = 'important_words'
VECTORIZER_FILE = 'feature_mapper'
ENSEMBLER_FILE = 'ensembler'
ENCODER_FILE = 'label_encoder'
CLASSIFIER_FILE='classifiers'
# DELIMITER="[|][|]"
DELIMITER="[,]"
MEDICAL_CONCEPTS = 'Medical Concepts for word2vec.csv'
ABBREVIATIONS_FILE = 'Abbreviations_ReverseMapping.csv'
LABELED_FILE = 'AllTrainingFinal.csv'

TRAINING_FILE = 'Training_R9.csv'
TEST_FILE = 'Testing_R9.csv'
UNLABELED_FILE = 'NormalizedOutput.csv'
ONE_DAY_DATA='OneDayData.csv'
VOCABULARY_TRAIN_FILE = 'vocabulary_train.csv'
VOCABULARY_TEST_FILE = 'vocabulary_test.csv'
VOCABULARY_UNLABELED_FILE = 'vocabulary_unlabeled.csv'
BOW_TRAIN_FILE = 'BOW_train_features.csv'
BOW_TEST_FILE = 'BOW_test_features.csv'
BOW_UNLABELED_FILE='BOW_unlabeled_features.csv'
MM_TRAIN_FILE= 'MM_train_features.csv'
MM_TEST_FILE='MM_test_features.csv'
MM_UNLABELED_FILE='MM_unlabeled_features.csv'
W2V_TRAIN_FILE='W2V_train_features.csv'
W2V_TEST_FILE='W2V_test_features.csv'
W2V_UNLABELED_FILE='W2V_unlabeled_features.csv'


COMBINED_ALL_TRAIN_FILE = 'combined_all_train.csv'
COMBINED_ALL_TEST_FILE = 'combined_all_test.csv'
COMBINED_ALL_UNLABELED_FILE = 'combined_all_unlabeled.csv'
METAMAP_CONCEPTS_FILE = 'Retained Semantic types.csv'
REMOVE_STOPWORDS = True
REMOVE_NUMBERS = False
BOW_MAX_FEATURES = 550
BOW_N_GRAM_RANGE = (1, 1)

# setup parameters for xgboost
XGB_PARAMS = {}
# use softmax multi-class classification
XGB_PARAMS['objective'] = 'multi:softprob' # 'multi:softmax'
# scale weight of positive examples
XGB_PARAMS['booster'] = 'gblinear'
XGB_PARAMS['eta'] = 0.05
XGB_PARAMS['max_depth'] = 6
XGB_PARAMS['silent'] = 1
XGB_PARAMS['nthread'] = 4
XGB_PARAMS['seed'] = 786
XGB_PARAMS['subsample'] = 0.7
XGB_PARAMS['colsample_bytree'] = 0.7
XGB_ROUNDS = 500
XGB_EARLY_STOPPING_ROUNDS = 10
STD_DEV_CUTOFF = 0.03
CORRELATION_CUTOFF = 1


ENCODING_TO_USE = None

syntactic_classifiers = ['Jaro', 'DLevenshtein']
classifiers = collections.OrderedDict()

# #TODO uncomment following 2 lines to use Jaro and DLevenshtein classifiers
classifiers['Jaro'] = JaroWrinklerClassifier()
classifiers['DLevenshtein'] = DamerauLevenshteinClassifier()

# classifiers['SVM'] = SVC(kernel='linear', probability=True)
# classifiers['Naive'] = MultinomialNB() # class_prior=class_prior
classifiers['Logistic_OvR'] = LogisticRegression(multi_class='ovr')
# classifiers['Randomforest'] = RandomForestClassifier(random_state=12246)

classifiers['DeepNN'] = Classifier(
    layers=[
        Layer("Tanh", units=209), #, pieces=2
        Layer("Softmax")],
    learning_rate=0.018,
    n_iter=25, random_state=786)

# dNN=Classifier(
#     layers=[
#         Layer("Sigmoid", units=100, pieces=2),
#         Layer("Softmax")],
#     learning_rate=0.001,
#     n_iter=25, random_state=786)
# classifiers['DeepNN'] =RandomizedSearchCV(dNN, param_distributions={
#     'learning_rate': stats.uniform(0.001, 0.05),
#     'hidden0__units': stats.randint(10,1000),
#     'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
# classifiers['kNN'] = KNeighborsClassifier(10)
# classifiers['LDA'] = LinearDiscriminantAnalysis()
# classifiers['XGBoost'] = (XGB_PARAMS, XGB_ROUNDS)

NO_EXTRA_FILES = 4
NUMBER_OF_MODELS = len(classifiers.keys())+NO_EXTRA_FILES
if os.path.exists('../' + MODELS_FOLDER_LEARNING+'/'+CLASSIFIER_FILE+'.pickle'):
    # print 'classifiers.pickle from MODELS_FOLDER_LEARNING'
    f = open('../' + MODELS_FOLDER_LEARNING+'/'+CLASSIFIER_FILE+'.pickle', 'rb')
    classifiersFinal = pickle.load(f)
    NUMBER_OF_MODELS = len(classifiersFinal)+NO_EXTRA_FILES
    f.close()
elif os.path.exists('../' + MODELS_FOLDER_ARCHIVED):
    # add check to check in archive
    dir_list = os.listdir('../'+MODELS_FOLDER_ARCHIVED+'/')
    if len(dir_list) > 0:
        dates_list = [datetime.strptime(date, DATETIME_FORMAT) for date in dir_list]
        index, value = max(enumerate(dates_list), key=operator.itemgetter(1))
        latest_archive = datetime.strftime(value,DATETIME_FORMAT)
        if os.path.exists('../' + MODELS_FOLDER_ARCHIVED+'/'+latest_archive+'/'+CLASSIFIER_FILE+'.pickle'):
            # print 'classifiers.pickle from MODELS_FOLDER_ARCHIVED'
            f = open('../' + MODELS_FOLDER_ARCHIVED+'/'+latest_archive+'/'+CLASSIFIER_FILE+'.pickle', 'rb')
            classifiersFinal = pickle.load(f)
            NUMBER_OF_MODELS = len(classifiersFinal)+NO_EXTRA_FILES
            f.close()

medical_concepts = collections.OrderedDict()
if os.path.exists('../' + WORD2VEC_FOLDER + '/' + MEDICAL_CONCEPTS):
    dfs_concepts = pd.read_csv('../' + WORD2VEC_FOLDER + '/' + MEDICAL_CONCEPTS)
    concepts = dfs_concepts['Concept']
    key_words = dfs_concepts['word2vec key words']
    for i, concept in enumerate(concepts):
        medical_concepts[concept.encode('ascii', 'ignore')] = [i.strip() for i in key_words[i].encode('ascii', 'ignore').split(',')]
else:
    print 'Medical Concepts file is not present in app\word2vec folder. It should be fetched from mercurial repository. '
    print 'Learning pipeline run was not successful. Exiting now......'
    sys.exit()

featureExtractionFunctions="combineW2VBOW"
combinedfiles='combined_bow_w2v_unlabeled.csv'
combinedfiles_test='combined_bow_w2v_test.csv'
combinedfiles_train='combined_bow_w2v_train.csv'
featureFunctions='combinedFeaturesW2VBOW'
# MODELS_FOLDER_LEARNING=OUTPUT_FOLDER # + '/BOW_W2V'
# METAMAP_INSTALLATION_FOLDER = 'C:/public_mm'