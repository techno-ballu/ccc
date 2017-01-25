# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 15:38:54 2016

@author: abzooba
"""

from __future__ import division

# from app.conf import config
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit

TARGET = 'CHRG_CLS'
folder = 'release10'
df_training = pd.read_csv('../' + folder + '/TrainingData.csv') #, delimiter='\t'
# df_testing = pd.read_csv('../' + folder + '/Test.csv')
# df_unlabelled = pd.read_csv('../' + folder + '/Test_unlabelled.csv')

print 'training samples=', len(df_training)

# # filter out unbundling & planbenefit reason codes:
# df_training = df_training.loc[(df_training[config.REASON_CODE] == 'Unbundling')]
# df_testing = df_testing.loc[(df_testing[config.REASON_CODE] == 'Unbundling')]
# df_unlabelled = df_unlabelled.loc[(df_unlabelled[config.REASON_CODE] == 'Unbundling')]
# # (df_training[config.REASON_CODE] == 'PlanBenefit')
# print 'unbundled training samples=', len(df_training)

# # clean up some categories based on value counts:
#
# # Class lab
# mask = (df_training[TARGET] == 'lab') | \
#        (df_training[TARGET] == 'lab ') | \
#        (df_training[TARGET] == 'Lab')
# df_training.loc[mask, TARGET] = 'Laboratory'
#
# # Class Routine Service
# mask = (df_training[TARGET] == 'routine services') | \
#        (df_training[TARGET] == 'Routine Services') | \
#        (df_training[TARGET] == 'rt service') | \
#        (df_training[TARGET] == 'RT Service') | \
#        (df_training[TARGET] == 'nursing service') | \
#        (df_training[TARGET] == 'respiratory') | \
#        (df_training[TARGET] == 'Service') | \
#        (df_training[TARGET] == 'Nursing')
# df_training.loc[mask, TARGET] = 'Routine Service'
#
# # Class Diluent/Irrigant
# mask = (df_training[TARGET] == 'diluent/irrigant') | \
#        (df_training[TARGET] == 'Irrigation') | \
#        (df_training[TARGET] == 'diluent_irrigant')
# df_training.loc[mask, TARGET] = 'Diluent Irrigant'
#
# # Class med/surg supply
# mask = (df_training[TARGET] == 'med_surg supply') | \
#        (df_training[TARGET] == 'Supply') | \
#        (df_training[TARGET] == 'med/surg supply')
# df_training.loc[mask, TARGET] = 'Medical Supply'
#
# # Pharmacy
# mask = (df_training[TARGET] == 'Pharmacy(need not classify)')
# df_training.loc[mask, TARGET] = 'Pharmacy'
#
# # Capital Equipment
# mask = (df_training[TARGET] == 'equipment')
# df_training.loc[mask, TARGET] = 'Capital Equipment'
#
# # Monitoring
# mask = (df_training[TARGET] == 'monitoring')
# df_training.loc[mask, TARGET] = 'Monitoring'
#
# # categories to be dropped
# drop_categories = ['-', 'ERROR DO NOT APPLY - REMOVE', 'Saline49Less', 'Blood', 'Education/Training' ]
# df_training = df_training.loc[~df_training[TARGET].isin(drop_categories)]
#
# # Fix classes in test set
# mask = (df_testing[TARGET] == 'monitoring')
# df_testing.loc[mask, TARGET] = 'Monitoring'
#
# mask = (df_testing[TARGET] == 'diluent/irrigate')
# df_testing.loc[mask, TARGET] = 'Diluent Irrigant'
#
# mask = (df_testing[TARGET] == 'med/surg supply')
# df_testing.loc[mask, TARGET] = 'Medical Supply'
#
# mask = (df_testing[TARGET] == 'lab') | (df_testing[TARGET] == 'Lab')
# df_testing.loc[mask, TARGET] = 'Laboratory'
#
# mask = (df_testing[TARGET] == 'routine service') | \
#        (df_testing[TARGET] == 'RT Service') | \
#        (df_testing[TARGET] == 'RT service')
# df_testing.loc[mask, TARGET] = 'Routine Service'
#
# mask = (df_testing[TARGET] == 'capital equipment')
# df_testing.loc[mask, TARGET] = 'Capital Equipment'

mask = (df_training[TARGET] == 'Diluent / flush / Irrigant')
df_training.loc[mask, TARGET] = 'Diluent / Flush / Irrigant'
print df_training[TARGET].value_counts()
# print df_testing[TARGET].value_counts()
print df_training.columns
# print df_testing.columns


y = df_training[TARGET].values

# spilt into stratified training & test sets
sss = StratifiedShuffleSplit(y, 1, 0.21, random_state=10002)
train_index, test_index = list(sss)[0]

df_train = df_training.iloc[train_index]
df_val = df_training.iloc[test_index]

train_count = len(df_train)
test_count = len(df_val)
# # display stats
# print '\nnumber of total samples = ', len(y)
# print df_claims_feats[TARGET].value_counts()
#
# print '\nnumber of training samples = ', len(df_train)
# print df_train[TARGET].value_counts()
#
# print '\nnumber of test samples = ', len(df_test)
# print df_test[TARGET].value_counts()

# save the training & test sets
df_train.to_csv('../' + folder + '/training_' + str(train_count) + '.csv', index=False)
df_val.to_csv('../' + folder + '/testing_' + str(test_count) + '.csv', index=False)
# df_unlabelled.to_csv('../' + folder + '/unlabelled.csv', index=False)