from __future__ import division
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.conf import appconfig
from app.core import data_preprocessor
from app.core import normalizer
from app.core import feature_mapper
from app.core import learner
from app.core import classifier
from datetime import datetime
from app.core import updateTables
import pandas as pd
import numpy as np
import glob
import time
import shutil
import tempfile
import logging
import collections
import pickle

start = time.time()
print (time.strftime("%H:%M:%S"))

# suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

target = appconfig.TARGET
data_folder = appconfig.DATA_MODELS_LEARNING
archived_folder = appconfig.MODELS_FOLDER_ARCHIVED
models_folder = appconfig.MODELS_FOLDER_LEARNING
training_file = appconfig.TRAINING_FILE
min_sample_size = appconfig.MIN_SAMPLE_SIZE
vectorizer_file = appconfig.VECTORIZER_FILE
ensembler_file = appconfig.ENSEMBLER_FILE
encoder_file = appconfig.ENCODER_FILE
classifier_file=appconfig.CLASSIFIER_FILE
text_feature = appconfig.TEXT_FEATURE
original_feature = appconfig.ORIGINAL_DESCRIPTION
remove_stopwords = appconfig.REMOVE_STOPWORDS
remove_numbers = appconfig.REMOVE_NUMBERS
classifiers = appconfig.classifiers
classifiers_final=collections.OrderedDict()
w2v_trainfile = appconfig.W2V_TRAIN_FILE
w2v_testfile = appconfig.W2V_TEST_FILE
vocab_trainfile = appconfig.VOCABULARY_TRAIN_FILE
vocab_testfile = appconfig.VOCABULARY_TEST_FILE
mmtrainfile=appconfig.MM_TRAIN_FILE
selected_features_file = appconfig.SELECTED_FEATURES_FILE
datetime_format = appconfig.DATETIME_FORMAT
classification_folder = appconfig.MODELS_FOLDER_CLASSIFICATION

def deleteOldArchives():
    logging.info(' : '+__file__+' : '+'In delete Old Archives')
    max_archive_folders = appconfig.MAX_ARCHIVE_FOLDERS
    dir_list = os.listdir('../'+archived_folder+'/')
    # print dir_list

    if len(dir_list) >= max_archive_folders:
        logging.debug(' : '+__file__+' : '+'deleting old archives...')

        dates_list = [datetime.strptime(date, datetime_format) for date in dir_list]
        dates_list.sort()
        sorted_folders = [datetime.strftime(date, datetime_format) for date in dates_list]
        # print sorted_folders

        for index,folder in enumerate(sorted_folders):
            if index == len(dir_list)-max_archive_folders+1:
                break
            logging.debug(' : '+__file__+' : '+'deleting archive folder:' + folder)

            tmp = tempfile.mktemp(dir=os.path.dirname('../'+archived_folder+'/'+folder))
            shutil.move('../'+archived_folder+'/'+folder, '../'+archived_folder+'/' +tmp)
            shutil.rmtree('../'+archived_folder+'/' + tmp)


def manage_folders():
    logging.info(' : '+__file__+' : '+'In manage_folders method')
    if not os.path.exists('../' + classification_folder):
        # create the models folder if not exists
        os.makedirs('../' + classification_folder)

    if not os.path.exists('../' + data_folder):
        # create the models folder if not exists
        os.makedirs('../' + data_folder)

    if os.path.exists('../' + models_folder):
        models_empty = True
        try:
            os.rmdir('../' + models_folder)
            # log this print #done
            logging.info(' : '+__file__+' : '+'models folder is empty,no need to archive')
            print models_folder, 'is empty.'
        except OSError:
            models_empty = False
            logging.info(' : '+__file__+' : '+'models folder is not empty, archiving models..')
            print models_folder, 'is not empty.'

        if models_empty == False:
            learning_model_list = [s for s in os.listdir('../' + models_folder) if s.endswith('.pickle')]
            if len(learning_model_list) == appconfig.NUMBER_OF_MODELS:

                timestamp_folder = '../' + archived_folder + '/' + time.strftime(datetime_format)

                # we need to create new datetime archive
                if not os.path.exists('../' + archived_folder):
                    os.makedirs('../' + archived_folder)

                # delete old archive, retain only recent archives (appconfig.MAX_ARCHIVE_FOLDERS=10)
                try:
                    deleteOldArchives()
                except:
                    logging.debug(' : '+__file__+' : '+'Unable to delete old archives, Continuing...')

                # copy the models folder to archive folder
                shutil.copytree('../' + models_folder, timestamp_folder)

                # rename to a temp folder and delete it
                tmp = tempfile.mktemp(dir=os.path.dirname('../' + models_folder))
                # Rename the dir.
                shutil.move('../' + models_folder, '../' +tmp)
                # And delete it.
                shutil.rmtree('../' + tmp)
                # shutil.rmtree('../' + models_folder)
            else:
                tmp = tempfile.mktemp(dir=os.path.dirname('../' + models_folder))
                shutil.move('../' + models_folder, '../' +tmp)
                shutil.rmtree('../' + tmp)
                # for the_file in os.listdir('../' + models_folder):
                #     file_path = os.path.join('../' + models_folder, the_file)
                #     try:
                #         if os.path.isfile(file_path):
                #             os.unlink(file_path)
                #         #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                #     except Exception as e:
                #         print(e)

    os.makedirs('../' + models_folder)

def train_pipeline():

    starttrain=time.time()

    featureExtractionFn = appconfig.featureExtractionFunctions
    featureFn = appconfig.featureFunctions
    features_train_file = appconfig.combinedfiles_train
    features_test_file = appconfig.combinedfiles_test

    # Load the data preprocessor to load, filter, clean & save the data.

    startLoad=time.time()

    preprocessor = data_preprocessor.DataPreprocessor(target, data_folder, models_folder, training_file, None) # , config.UNLABELLED_FILE
    logging.info(' : '+__file__+' : '+'Data preprocessor initialized.')
    preprocessor.loadData()
    data = preprocessor.saveData()
    print "Time taken in loading and saving training data:"
    print time.time()-startLoad

    startUpsample=time.time()
    data = preprocessor.upsampleMinorChargeClasses(min_sample_size)
    logging.info(' : '+__file__+' : '+'Upsampling(if needed) done.')
    print "Time taken in upsampling:"
    print time.time()-startUpsample

    startSplit=time.time()
    # move the split size and step size to config # done
    random_state = 786
    test_size = appconfig.MIN_TEST_TRAIN_SPLIT
    step_size = appconfig.SPLIT_STEP_SIZE
    len_y = len(data['training'])
    # df_training, df_testing = preprocessor.stratifiedSplitData(data['training'], test_size, target)
    df_training, df_testing = preprocessor.shuffleSplitData(len_y, test_size, data['training'], random_state)
    logging.debug(' : '+__file__+' : training size = ' + str(len(df_training)))
    logging.debug(' : '+__file__+' : testing size = ' + str(len(df_testing)))
    logging.debug(' : '+__file__+' : training targets min value_counts = '+str(min(df_training[target].value_counts())))
    logging.debug(' : '+__file__+' : testing targets min value_counts = '+str(min(df_testing[target].value_counts())))
    # while min(min(df_testing[target].value_counts()),min(df_training[target].value_counts()))<=1:
    # while min(df_training[target].value_counts())<=1:
    while len(df_testing[target].unique()) == 1:
        logging.debug(' : '+__file__+' : while : training targets min value_counts = '+str(min(df_training[target].value_counts())))
        logging.debug(' : '+__file__+' : while : testing targets min value_counts = '+str(min(df_testing[target].value_counts())))
        random_state += step_size
        # df_training, df_testing = preprocessor.stratifiedSplitData(data['training'], Split_min, target)
        df_training, df_testing = preprocessor.shuffleSplitData(len_y, test_size, data['training'], random_state)
        if random_state > (random_state + appconfig.MAX_STEP_SIZE):
            # logging.warning(' : '+__file__+' : '+'Number of distinct charge classes are too large to be handled by this application.')
            logging.warning(' : '+__file__+' : '+'Test data has just one charge class!')
            break

    print "Time taken in splitting test from training:"
    print time.time()-startSplit
    # print df_training[target].value_counts(), df_testing[target].value_counts()

    # backup Normalized description
    normalized_desc_train = pd.DataFrame(df_training[text_feature])
    normalized_desc_test = pd.DataFrame(df_testing[text_feature])

    # create Original descriptions copy
    y_ori = np.array(data['training'][target].values)
    original_desc = np.array(data['training'][original_feature].values)
    original_desc_train = np.array(df_training[original_feature].values)
    original_desc_test = np.array(df_testing[original_feature].values)

    # clean claim lines using sentence cleaner in normalizer.
    startNorm=time.time()
    logging.info(' : '+__file__+' : '+'Normalizer intialized.')
    data_normalizer = normalizer.Normalizer()
    logging.info(' : '+__file__+' : '+'After init')
    df_training = data_normalizer.normalize(df_training, text_feature, original_feature, '../' + models_folder + '/' + vocab_trainfile)
    logging.info(' : '+__file__+' : '+'training norm')
    df_testing = data_normalizer.normalize(df_testing, text_feature, original_feature, '../' + models_folder + '/' + vocab_testfile)
    logging.info(' : '+__file__+' : '+'testting norm')
    df_comparison = df_testing[[original_feature, text_feature, target]]
    logging.info(' : '+__file__+' : '+'Basic cleaninng done. ')
    print "Time taken in normalization inside python:"
    print time.time()-startNorm


    # instantiate the Feature Mapper
    mapper = feature_mapper.FeatureMapper()
    logging.info(' : '+__file__+' : '+'FeatureMapper instantiated')
    # train the CountVectorizer model
    startVector=time.time()
    cleaned_claims_train = np.array(df_training[text_feature].values)
    mapper.trainVectorizer(cleaned_claims_train)
    print "Time taken in training countVectorizer:"
    print time.time()-startVector

    startFeatVector=time.time()
    df_train_vectors = getattr(mapper,featureExtractionFn)(df_training, features_train_file,w2v_trainfile)
    df_testing_vectors = getattr(mapper,featureExtractionFn)(df_testing, features_test_file,w2v_testfile)
    logging.debug(' : '+__file__+' : '+'Training set shape.'+ str(df_train_vectors.shape))
    logging.debug(' : '+__file__+' : '+'Test set shape.'+ str(df_testing_vectors.shape))
    logging.info(' : '+__file__+' : '+'Feature extraction done.')
    logging.debug(' : '+__file__+' : '+'Time taken in extracting feature vectors:'+ str(time.time()-startFeatVector))
    print "Time taken in extracting feature vectors:"
    print time.time()-startFeatVector

    features_all = getattr(mapper, featureFn)()
    train_vectors, testing_vectors, features = mapper.removeOnlyZeroVariance(df_train_vectors, df_testing_vectors, features_all, features_train_file)
    print train_vectors.shape, testing_vectors.shape
    logging.debug(' : '+__file__+' : '+'Training set shape after remove zero variance.'+ str(train_vectors.shape))
    logging.debug(' : '+__file__+' : '+'test set shape  after remove zero variance.'+ str(testing_vectors.shape))

    # save the Vectorizer object to disk
    startSavevector=time.time()
    mapper.saveFeatureMapper(vectorizer_file)

    # save the selected features to disk
    mapper.saveSelectedFeatures(selected_features_file, features)
    print "Time taken in saving Vectorizer object and delected features:"
    print time.time()-startSavevector

    y = df_training[target].values
    classes_ = df_training[target].unique()
    y_test = df_testing[target].values
    df_training[text_feature] = normalized_desc_train

    learning = learner.LearningEngine(classifiers, train_vectors, testing_vectors, y, y_test,
                                      df_comparison, features, original_desc, original_desc_test, y_ori)
    logging.info(' : '+__file__+' : '+'LearningEngine instantiated.')
    # learning.crossValidateClassifiers()
    df_comparison, imp_words = learning.trainClassifiers()
    logging.info(' : '+__file__+' : '+'Classifiers trained.')

    # probColumns=['Prob_'+clf_name for clf_name in classifiers]
    # chargeClassColumns=['Predicted_Class_'+clf_name for clf_name in classifiers]

    for clf_name in classifiers:
        logging.info(' : '+__file__+' : checking : ' + clf_name)
        try:
            chrg_class=len(df_comparison['Predicted_Class_'+clf_name].unique())
            if chrg_class>1 and df_comparison['Prob_'+clf_name].isnull().values.any()==False:
                classifiers_final[clf_name]=classifiers[clf_name]
                logging.debug(' : '+__file__+' : '+ str(clf_name) +' was used as an learning algorithm')
            else:
                logging.warning(' : '+__file__+' : WARNING!! Algorithm '+ str(clf_name) +' was not used in learning pipeline')
                try:
                    os.remove('../' + models_folder+'/'+clf_name +'.pickle')
                except OSError:
                    pass
        except Exception as e:
            logging.error(e)
            logging.warning(' : '+__file__+' :  WARNING!! Algorithm '+ str(clf_name) +' was not used in learning pipeline')
            try:
                os.remove('../' + models_folder+'/'+clf_name +'.pickle')
            except OSError:
                pass


    classifiersFile=open('../' + models_folder+'/'+classifier_file +'.pickle','wb')
    pickle.dump(np.array(classifiers_final.keys()),classifiersFile)
    classifiersFile.close()

    startvoting=time.time()
    voter = learner.Voter(df_comparison, classifiers_final)
    df_comparison = voter.votingPredictions()
    logging.info(' : '+__file__+' : '+'Voting done.')
    print "Time taken in voting:"
    print time.time()-startvoting

    df_comparison.to_csv('../' + models_folder + '/df_comparison.csv', index=True)

    # # TODO uncomment files deletion
    # # delete temporary csv files:
    # for csv_file_path in glob.glob('../' + models_folder + "/*.csv"):
    #     os.remove(csv_file_path)
    #
    # for csv_file_path in glob.glob('../' + data_folder + "/*.csv"):
    #     os.remove(csv_file_path)
    # logging.info(' : '+__file__+' : '+'Removed temporary csv files.')
    # print '-'*100, '\n'

    training_date = str(datetime.now())
    train_rec = len(df_training)
    test_rec = len(df_testing)
    no_of_chrg_cls = len(classes_)
    # TODO fix issue!
    onl_retro_ratio = -9.9
    if len(df_training[df_training['Flag']=='RETROSPECTIVE']) != 0:
        onl_retro_ratio = float(len(df_training[df_training['Flag']=='ONLINE'])/len(df_training[df_training['Flag']=='RETROSPECTIVE']))
    bowFeat = len(mapper.bowFeatures())
    w2vFeat = len(mapper.w2vFeatures())
    std_cutoff = appconfig.STD_DEV_CUTOFF
    correlation_cutoff=appconfig.CORRELATION_CUTOFF
    no_of_selFeat = len(features)
    classif_names = ";".join(classifiers_final.keys())
    metadata = [training_date, train_rec, test_rec, no_of_chrg_cls,
                onl_retro_ratio,
                bowFeat, w2vFeat,
                no_of_selFeat, classif_names, std_cutoff,correlation_cutoff]
    print "Time taken in training"
    print time.time()-starttrain
    return metadata

if __name__ == "__main__":
    train_pipeline()
    print (time.strftime("%H:%M:%S"))
    done = time.time()
    elapsed = done - start
    print 'elapsed = ', elapsed
