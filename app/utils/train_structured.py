import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from app.conf import appconfig
from app.core import normalizer
from app.core import feature_mapper
from app.core import learner
import pickle
import collections

##########################################################################
# Configurable constants
REV_CODE = 'Rev_Code_Name'
NORM_BILL = 'norm_billed_amount'
UNITS = 'units'
BILL_CATS = 'billed_categories'
UNITS_CATS = 'units_categories'
appconfig.TARGET = TARGET = 'CHRG_CLS'
appconfig.ORIGINAL_DESCRIPTION = ORIG_FEATURE = 'cleaned_descrip'
appconfig.TEXT_FEATURE = NORM_FEATURE = 'NORM_DESC'
models_folder = 'release10'
classifiers = appconfig.classifiers
##########################################################################

# load the (joined with structured and aggregated) training data
# df_train = pd.read_csv('../' + models_folder + '/df_train_struc_joined_latest.tsv', sep='\t')
df_train = pd.read_csv('../' + models_folder + '/structured_data.txt', sep='\t')
print df_train.columns
print 'columns in training data =', ', '.join(df_train.columns.values)
print 'shape of train data =', df_train.shape
print 'number of distinct descriptions = ', len(df_train[ORIG_FEATURE].unique())

max_billed = df_train[NORM_BILL].max()
bins = [-10, 0, 4, 17, 41, 61, 92, 287, 641, 997, 2496, 9320, max_billed]
# group_names = ['Zero $ value', 'Low $ value', 'Okay $ value', 'High $ value']
df_train[BILL_CATS] = pd.cut(df_train[NORM_BILL], bins) #, labels=group_names
# print df_train['billed_categories'].value_counts()

max_units = df_train[UNITS].max()
# bins = [0, 1.25, 1.75, 2.75, 6.75, 7.75, 24.75, 103, max_units]
bins = [0, 1.00, max_units]
group_names = ['Units <= 1', 'Units > 1']
df_train[UNITS_CATS] = pd.cut(df_train[UNITS], bins, labels=group_names) #
# print df_train['units_categories'].value_counts()

# clean the outlier charge class
df_train.loc[df_train[TARGET] == 'Diluent / flush / Irrigant', TARGET] = 'Diluent / Flush / Irrigant'
df_train.loc[df_train[REV_CODE] == 'Clinic', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Nuclear Medicine', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Ambulance', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Medical Equipment', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Oncology', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Ambulatory Surgical', REV_CODE] = 'Others'
df_train.loc[df_train[REV_CODE] == 'Home Health', REV_CODE] = 'Others'
df_train.to_csv('../' + models_folder + '/structured_data.tsv', index=False, sep='\t')
# # print df_train[REV_CODE].value_counts()
#
# structured_cols = []
#
# # Adding Rev Code Group dummy cloumns
# df_rev_dummies = pd.get_dummies(df_train[REV_CODE], drop_first=True)
# revcode_dummy_cols = list(df_rev_dummies.columns.values)
# print 'shape of revcode dummies =', df_rev_dummies.shape
# print 'columns in revcode dummies =', ', '.join(revcode_dummy_cols)
# df_train = pd.concat([df_train, df_rev_dummies], axis=1)
# structured_cols.extend(revcode_dummy_cols)
#
# # Adding Norm billed amount dummy cloumns
# df_bill_dummies = pd.get_dummies(df_train[BILL_CATS], drop_first=True)
# bill_dummy_cols = list(df_bill_dummies.columns.values)
# print 'shape of billed dummies =', df_bill_dummies.shape
# print 'columns in billed dummies =', ', '.join(bill_dummy_cols)
# df_train = pd.concat([df_train, df_bill_dummies], axis=1)
# structured_cols.extend(bill_dummy_cols)
#
# # Adding units dummy cloumns
# df_units_dummies = pd.get_dummies(df_train[UNITS_CATS], drop_first=True)
# units_dummy_cols = list(df_units_dummies.columns.values)
# print 'shape of units dummies =', df_units_dummies.shape
# print 'columns in units dummies =', ', '.join(units_dummy_cols)
# df_train = pd.concat([df_train, df_units_dummies], axis=1)
# structured_cols.extend(units_dummy_cols)
#
# print 'shape of train data after adding dummies =', df_train.shape
# print 'columns added = ', ', '.join(structured_cols)
#
# # split the training data for a stratified 10 fold cross validation
# y = df_train[TARGET]
#
# # # Stratified split to be done on unique descriptions
# # df_train_unique = df_train.drop_duplicates(subset = [ORIG_FEATURE, TARGET], inplace = False)
# # y_unique = df_train_unique[TARGET]
# # print 'shape of train data after taking uniques only =', df_train_unique.shape
#
# # Stratified sampling done once, Now just need to read training and test descriptions from file
# df_training_all = pd.read_excel('../' + models_folder + '/trainingAll.xlsx')
# df_training_unique = df_training_all[df_training_all['Flag'] == 'TRAIN'][ORIG_FEATURE]
# df_testing_unique = df_training_all[df_training_all['Flag'] == 'TEST'][ORIG_FEATURE]
# print 'shape of train data after taking uniques only =', df_training_unique.shape
# print 'shape of test data after taking uniques only =', df_testing_unique.shape
#
# # # sample the data to do quick runs
# # sss = StratifiedShuffleSplit(y, 1, test_size=0.9, random_state=0)
# # sample_idx, throw_idx = list(sss)[0]
# # df_train = df_train.iloc[sample_idx]
# # print 'shape of train data after sampling =', df_train.shape
# # y = df_train[TARGET]
#
# # check frequency table of target
# print y.value_counts();print
#
#
# # # Stratified K-fold split
# # skf = StratifiedKFold(y=y_unique, n_folds=5, shuffle=True, random_state=10449)
#
# accuracies = []
# fold_counter = 1
# print '='*10, 'fold ', fold_counter, '='*10
#
#
# df_training = df_train.merge(pd.DataFrame(df_training_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)
# df_testing = df_train.merge(pd.DataFrame(df_testing_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)
#
# # df_training = df_train.iloc[train_index]
# # df_testing = df_train.iloc[test_index]
#
# print 'training data shape after split = ', df_training.shape
# print 'testing data shape after split = ', df_testing.shape
#
# # Cleaning descriptions
# data_normalizer = normalizer.Normalizer()
# df_training = data_normalizer.normalize(df_training, NORM_FEATURE, ORIG_FEATURE, None)
# df_testing = data_normalizer.normalize(df_testing, NORM_FEATURE, ORIG_FEATURE, None)
#
# df_comparison = df_testing[[ORIG_FEATURE, NORM_FEATURE, TARGET]]
# original_desc_train = np.array(df_training[ORIG_FEATURE].values)
# original_desc_test = np.array(df_testing[ORIG_FEATURE].values)
#
#
# mapper = feature_mapper.FeatureMapper()
#
# # train the CountVectorizer model
# cleaned_claims_train = np.array(df_training[NORM_FEATURE].values)
# mapper.trainVectorizer(cleaned_claims_train)
#
# df_train_vectors = mapper.combineW2VBOW(df_training)
# df_testing_vectors = mapper.combineW2VBOW(df_testing)
#
# features_all = mapper.combinedFeaturesW2VBOW()
# train_vectors, testing_vectors, features = mapper.removeOnlyZeroVariance(df_train_vectors, df_testing_vectors)
#
# print 'training data shape after feature extraction = ', train_vectors.shape
# print 'testing data shape after feature extraction = ', testing_vectors.shape
#
# # TODO add structured columns to vectors
# structured_vectors_train = df_training[structured_cols].values
# structured_vectors_test = df_testing[structured_cols].values
# train_vectors = np.hstack([train_vectors, structured_vectors_train])
# testing_vectors = np.hstack([testing_vectors, structured_vectors_test])
#
# print 'training data shape after adding structured = ', train_vectors.shape
# print 'testing data shape after adding structured = ', testing_vectors.shape
#
# y_train = df_training[TARGET].values
# classes_ = df_training[TARGET].unique()
# y_test = df_testing[TARGET].values
#
# learning = learner.LearningEngine(classifiers, train_vectors, testing_vectors, y_train, y_test,
#                               df_comparison, features, original_desc_train, original_desc_test, y_train)
# df_comparison, imp_words = learning.trainClassifiers()
#
# # ============================= First time ===============================
# # #Syntactic classifiers' predictions to pickle (one time)
# # col_names = ['cleaned_descrip','Predicted_Class_Jaro','Prob_Jaro'	,
# #              'Predicted_Class_DLevenshtein'	,'Prob_DLevenshtein']
# #
# # f = open('../' + models_folder + '/syntactic_predictions.pickle', 'wb')
# # pickle.dump(df_comparison[col_names], f)
# # f.close()
# # classifiers_copy = classifiers
# # ============================= First time ===============================
#
# # ============================= Other times ===============================
# # Load syntactic classifiers' predictions from pickle file
# f = open('../' + models_folder + '/syntactic_predictions.pickle', 'rb')
# df_syntactic = pickle.load(f)
# f.close()
# df_syntactic = df_syntactic.drop_duplicates(subset = [ORIG_FEATURE], inplace = False)
# df_comparison = df_comparison.merge(df_syntactic, on = ORIG_FEATURE)
# classifiers_copy = collections.OrderedDict({'Jaro':'','DLevenshtein':'','Logistic_OvR':'','DeepNN':''})
# # ============================= Other times ===============================
#
#
# voter = learner.Voter(df_comparison, classifiers_copy)
# df_comparison = voter.votingPredictions()
#
#
# df_comparison.loc[(df_comparison['Prob_Jaro'] >= 0.8) & (df_comparison['Prob_Jaro'] < 0.85), 'Prob_Jaro'] = 0.85
# df_comparison.loc[(df_comparison['Prob_Logistic_OvR'] >= 0.8) & (df_comparison['Prob_Logistic_OvR'] < 0.9), 'Prob_Logistic_OvR'] = 0.96
# df_comparison.loc[(df_comparison['Prob_Logistic_OvR'] >= 0.7) & (df_comparison['Prob_Logistic_OvR'] < 0.8), 'Prob_Logistic_OvR'] = 0.92
# df_comparison.loc[(df_comparison['Prob_Logistic_OvR'] >= 0.6) & (df_comparison['Prob_Logistic_OvR'] < 0.7), 'Prob_Logistic_OvR'] = 0.76
# df_comparison.loc[(df_comparison['Prob_DLevenshtein'] >= 0.8) & (df_comparison['Prob_DLevenshtein'] < 0.9), 'Prob_DLevenshtein'] = 0.96
# df_comparison.loc[(df_comparison['Prob_DLevenshtein'] >= 0.7) & (df_comparison['Prob_DLevenshtein'] < 0.8), 'Prob_DLevenshtein'] = 0.93
# df_comparison.loc[(df_comparison['Prob_DLevenshtein'] >= 0.6) & (df_comparison['Prob_DLevenshtein'] < 0.7), 'Prob_DLevenshtein'] = 0.88
# df_comparison.loc[(df_comparison['Prob_DLevenshtein'] >= 0.5) & (df_comparison['Prob_DLevenshtein'] < 0.6), 'Prob_DLevenshtein'] = 0.81
#
# confidence = learner.ConfidenceCalculation(classifiers_copy, df_comparison)
# df_comparison = confidence.calculateByBoosting()
# y_pred = df_comparison['Voted Output'].values
#
# df_comparison.to_csv('../' + models_folder + '/df_comparison_RG_baseline_1.tsv', index=False)
# print 'accuracy =', accuracy_score(y_test, y_pred)
# accuracies.append(accuracy_score(y_test, y_pred))
# print
# fold_counter += 1
# print 'Overall accuracy =', np.mean(accuracies)
#
#
#
#
#
# # for train_index, test_index in skf:
# #     print '='*10, 'fold ', fold_counter, '='*10
# #
# #     df_training_unique = df_train_unique.iloc[train_index][ORIG_FEATURE]
# #     df_testing_unique = df_train_unique.iloc[test_index][ORIG_FEATURE]
# #     df_training = df_train.merge(pd.DataFrame(df_training_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)
# #     df_testing = df_train.merge(pd.DataFrame(df_testing_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)
# #
# #     # df_training = df_train.iloc[train_index]
# #     # df_testing = df_train.iloc[test_index]
# #
# #     print 'training data shape after split = ', df_training.shape
# #     print 'testing data shape after split = ', df_testing.shape
# #
# #     # Cleaning descriptions
# #     data_normalizer = normalizer.Normalizer()
# #     df_training = data_normalizer.normalize(df_training, NORM_FEATURE, ORIG_FEATURE, None)
# #     df_testing = data_normalizer.normalize(df_testing, NORM_FEATURE, ORIG_FEATURE, None)
# #
# #     df_comparison = df_testing[[ORIG_FEATURE, NORM_FEATURE, TARGET]]
# #     original_desc_train = np.array(df_training[ORIG_FEATURE].values)
# #     original_desc_test = np.array(df_testing[ORIG_FEATURE].values)
# #
# #
# #     mapper = feature_mapper.FeatureMapper()
# #
# #     # train the CountVectorizer model
# #     cleaned_claims_train = np.array(df_training[NORM_FEATURE].values)
# #     mapper.trainVectorizer(cleaned_claims_train)
# #
# #     df_train_vectors = mapper.combineW2VBOW(df_training)
# #     df_testing_vectors = mapper.combineW2VBOW(df_testing)
# #
# #     features_all = mapper.combinedFeaturesW2VBOW()
# #     train_vectors, testing_vectors, features = mapper.removeOnlyZeroVariance(df_train_vectors, df_testing_vectors)
# #
# #     print 'training data shape after feature extraction = ', train_vectors.shape
# #     print 'testing data shape after feature extraction = ', testing_vectors.shape
# #
# #     # # TODO add structured columns to vectors
# #     # structured_vectors_train = df_training[revcode_dummy_cols].values
# #     # structured_vectors_test = df_testing[revcode_dummy_cols].values
# #     # train_vectors = np.hstack([train_vectors, structured_vectors_train])
# #     # testing_vectors = np.hstack([testing_vectors, structured_vectors_test])
# #
# #     print 'training data shape after adding structured = ', train_vectors.shape
# #     print 'testing data shape after adding structured = ', testing_vectors.shape
# #
# #     y_train = df_training[TARGET].values
# #     classes_ = df_training[TARGET].unique()
# #     y_test = df_testing[TARGET].values
# #
# #     learning = learner.LearningEngine(classifiers, train_vectors, testing_vectors, y_train, y_test,
# #                                       df_comparison, features, original_desc_train, original_desc_test, y_train)
# #     df_comparison, imp_words = learning.trainClassifiers()
# #
# #     voter = learner.Voter(df_comparison, classifiers)
# #     df_comparison = voter.votingPredictions()
# #
# #     confidence = learner.ConfidenceCalculation(classifiers, df_comparison)
# #     df_comparison = confidence.calculateByBoosting()
# #     y_pred = df_comparison['Voted Output'].values
# #
# #     df_comparison.to_csv('../' + models_folder + '/df_comparison_RG_baseline.csv', index=True)
# #     print 'accuracy =', accuracy_score(y_test, y_pred)
# #     accuracies.append(accuracy_score(y_test, y_pred))
# #     print
# #     fold_counter += 1
# #
# #     break

