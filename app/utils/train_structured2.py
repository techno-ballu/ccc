import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from app.conf import appconfig
from app.core import normalizer
from app.core import feature_mapper
from app.core import learner

##########################################################################
# Configurable constants
REV_CODE = 'rev_code'
appconfig.TARGET = TARGET = 'CHRG_CLS'
appconfig.ORIGINAL_DESCRIPTION = ORIG_FEATURE = 'cleaned_descrip'
appconfig.TEXT_FEATURE = NORM_FEATURE = 'NORM_DESC'
models_folder = 'release10'
classifiers = appconfig.classifiers
##########################################################################

# load the (joined with structured and aggregated) training data
df_train = pd.read_csv('../' + models_folder + '/df_train_struc_joined_revcode_level.csv', sep='\t')
# df_test = pd.read_csv('../' + models_folder + '/test_predictions.csv')
print 'shape of train data =', df_train.shape
print 'number of distinct descriptions in training data = ', len(df_train[ORIG_FEATURE].unique())
print 'columns in training data =', ', '.join(df_train.columns.values)
# print 'shape of test data =', df_test.shape
# print 'number of distinct descriptions in test data = ', len(df_test[ORIG_FEATURE].unique())
# print 'columns in test data =', ', '.join(df_test.columns.values)

df_train['temp'] = 1.0
df_grouped = pd.pivot_table(df_train, index=['rev_code'], columns=['CHRG_CLS'], values='temp', aggfunc=np.sum)
# df_grouped['Dominant CC'] = df_grouped.idxmax(axis=1)
df_grouped_transpose = pd.DataFrame(df_grouped).T
# rslt = pd.DataFrame(np.zeros((0,2)), columns=['Dominant CC1','Dominant CC2'])
dominant1 = []
dominant2 = []
for col in df_grouped_transpose.columns:
    top2 = df_grouped_transpose.nlargest(2, col).index.tolist()
    if len(top2) < 2:
        top2.append('')
    dominant1.append(top2[0])
    dominant2.append(top2[1])
    # df1row = pd.DataFrame(top2, index=['Dominant CC1','Dominant CC2']).T
    # rslt = pd.concat([rslt, df1row], axis=0)

df_grouped['Dominant CC1'] = dominant1
df_grouped['Dominant CC2'] = dominant2
df_grouped['# of descriptions'] = df_grouped.sum(axis=1)
percent_cc1 = []
percent_cc2 = []
for index, row in df_grouped.iterrows():
    percent_cc1.append( (row[row['Dominant CC1']] / row['# of descriptions']) * 100)
    if row['Dominant CC2'] != '':
        percent_cc2.append( (row[row['Dominant CC2']] / row['# of descriptions']) * 100)
    else:
        percent_cc2.append(0)
df_grouped['% Dominance1'] = percent_cc1
df_grouped['% Dominance2'] = percent_cc2
df_grouped.insert(0, 'rev_code', df_grouped.index)
df_grouped = df_grouped[['rev_code', 'Dominant CC1', 'Dominant CC2', '# of descriptions', '% Dominance1',
                         '% Dominance2']]

df_grouped.to_csv('../' + models_folder + '/rev_code_chrgcls_pivot.csv', encoding='utf_8', index=False)
df_train = df_train.merge(df_grouped, 'left', 'rev_code')
df_train = df_train.drop('temp', 1)
df_train.to_csv('../' + models_folder + '/train_merged_rev_code_chrgcls_pivot.tsv', encoding='utf_8', index=False, sep='\t')

# df_test = df_test.merge(df_grouped, 'left', 'rev_code')
# df_test.to_csv('../' + models_folder + '/test_predictions.tsv', encoding='utf_8', index=False, sep='\t')


# clean the outlier charge class
df_train.loc[df_train[TARGET] == 'Diluent / flush / Irrigant', TARGET] = 'Diluent / Flush / Irrigant'

print df_train[REV_CODE].value_counts()

# structured_cols = []
# structured_cols.append(REV_CODE)
# structured_cols.append('norm_billed_amount')
# revcode_dummy_cols.append('units')
# print 'columns in structured =', ', '.join(structured_cols)

# split the training data for a stratified 10 fold cross validation
y = df_train[TARGET]

# Stratified split to be done on unique descriptions
df_train_unique = df_train.drop_duplicates(subset = [ORIG_FEATURE, TARGET], inplace = False)
y_unique = df_train_unique[TARGET]

# check frequency table of target
print y_unique.value_counts();print

# Stratified K-fold split
skf = StratifiedKFold(y=y_unique, n_folds=5, shuffle=True, random_state=10449)

accuracies = []
fold_counter = 1
for train_index, test_index in skf:
    print '='*10, 'fold ', fold_counter, '='*10

    df_training_unique = df_train_unique.iloc[train_index][ORIG_FEATURE]
    df_testing_unique = df_train_unique.iloc[test_index][ORIG_FEATURE]
    df_training = df_train.merge(pd.DataFrame(df_training_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)
    df_testing = df_train.merge(pd.DataFrame(df_testing_unique, columns = [ORIG_FEATURE]), on = ORIG_FEATURE)

    # df_training = df_train.iloc[train_index]
    # df_testing = df_train.iloc[test_index]

    print 'training data shape after split = ', df_training.shape
    print 'testing data shape after split = ', df_testing.shape

    # Cleaning descriptions
    data_normalizer = normalizer.Normalizer()
    df_training = data_normalizer.normalize(df_training, NORM_FEATURE, ORIG_FEATURE, None)
    df_testing = data_normalizer.normalize(df_testing, NORM_FEATURE, ORIG_FEATURE, None)

    df_comparison = df_testing[[ORIG_FEATURE, NORM_FEATURE, REV_CODE, TARGET, 'Dominant CC1', 'Dominant CC2', '# of descriptions', '% Dominance1', '% Dominance2']]
    original_desc_train = np.array(df_training[ORIG_FEATURE].values)
    original_desc_test = np.array(df_testing[ORIG_FEATURE].values)

    mapper = feature_mapper.FeatureMapper()

    # train the CountVectorizer model
    cleaned_claims_train = np.array(df_training[NORM_FEATURE].values)
    mapper.trainVectorizer(cleaned_claims_train)

    df_train_vectors = mapper.combineW2VBOW(df_training)
    df_testing_vectors = mapper.combineW2VBOW(df_testing)

    features_all = mapper.combinedFeaturesW2VBOW()
    train_vectors, testing_vectors, features = mapper.removeOnlyZeroVariance(df_train_vectors, df_testing_vectors)

    print 'training data shape after feature extraction = ', train_vectors.shape
    print 'testing data shape after feature extraction = ', testing_vectors.shape

    # # TODO add structured columns to vectors
    # structured_vectors_train = df_training[structured_cols].values
    # structured_vectors_test = df_testing[structured_cols].values
    # train_vectors = np.hstack([train_vectors, structured_vectors_train])
    # testing_vectors = np.hstack([testing_vectors, structured_vectors_test])
    #
    # print 'training data shape after adding structured = ', train_vectors.shape
    # print 'testing data shape after adding structured = ', testing_vectors.shape

    y_train = df_training[TARGET].values
    classes_ = df_training[TARGET].unique()
    y_test = df_testing[TARGET].values

    learning = learner.LearningEngine(classifiers, train_vectors, testing_vectors, y_train, y_test,
                                      df_comparison, features, original_desc_train, original_desc_test, y_train)
    df_comparison, imp_words = learning.trainClassifiers()

    voter = learner.Voter(df_comparison, classifiers)
    df_comparison = voter.votingPredictions()

    confidence = learner.ConfidenceCalculation(classifiers, df_comparison)
    df_comparison = confidence.calculateByBoosting()
    y_pred = df_comparison['Voted Output'].values

    df_comparison.loc[(df_comparison['Voted Output'] == df_comparison['Dominant CC1']) & (df_comparison['Confidence Score'] > 0.6), 'Confidence Score'] = 0.95

    df_comparison.to_csv('../' + models_folder + '/df_comparison.csv', index=True)
    print 'accuracy =', accuracy_score(y_test, y_pred)
    accuracies.append(accuracy_score(y_test, y_pred))
    print
    fold_counter += 1

    break

print 'Overall accuracy =', np.mean(accuracies)