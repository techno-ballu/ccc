import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import graphviz
from sklearn import tree

models_folder = 'release10'
TARGET = 'CHRG_CLS'
STRUCTURED = 'units'

df_structured = pd.read_csv('../' + models_folder + '/df_train_struc_joined_latest.tsv', sep='\t')
print df_structured.columns
print df_structured.shape

# clean the outlier charge class
df_structured.loc[df_structured[TARGET] == 'Diluent / flush / Irrigant', TARGET] = 'Diluent / Flush / Irrigant'
df_structured = df_structured[[STRUCTURED, TARGET]]

# split the training data for a stratified 10 fold cross validation
x = df_structured[STRUCTURED]
x = x[:, None]
y = df_structured[TARGET]

print x.shape, y.shape

clf = DecisionTreeClassifier(random_state=1007389, max_depth=4)
clf.fit(x, y)

with open('../' + models_folder + '/tree.dot', 'w') as dotfile:
    tree.export_graphviz(clf, dotfile, feature_names = [STRUCTURED])

# # Stratified split to be done on unique descriptions
#
# # sample the data to do quick runs
# sss = StratifiedShuffleSplit(y, 1, test_size=0.5, random_state=0)
# sample_idx, throw_idx = list(sss)[0]
# df_structured = df_structured.iloc[sample_idx]
# print 'shape of train data after sampling =', df_structured.shape
# y = df_structured[TARGET]
#
# # # check frequency table of target
# # print y.value_counts();print
#
# # Stratified K-fold split
# skf = StratifiedKFold(y=y, n_folds=5, shuffle=True, random_state=10449)
#
# accuracies = []
# fold_counter = 1
# for train_index, test_index in skf:
#     print '='*10, 'fold ', fold_counter, '='*10
#
#     df_training = df_structured.iloc[train_index]
#     df_testing = df_structured.iloc[test_index]
#
#     x_train = df_training[STRUCTURED].values
#     x_test = df_testing[STRUCTURED].values
#     y_train = df_training[TARGET].values
#     y_test = df_testing[TARGET].values
#     print x_train.shape, x_test.shape, y_train.shape, y_test.shape
#
#     clf = DecisionTreeClassifier(random_state=1007389)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#
#     print 'accuracy = ', accuracy_score(y_test, y_pred)
