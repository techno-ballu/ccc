import os
import pickle
import collections
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import precision_score, average_precision_score, \
    recall_score, accuracy_score, roc_curve

# folder = 'learning_models/models'
folder = 'classification_models'

# Read unlabelled file
df_unlabelled = pd.read_csv('../' + folder + '/df_unlabelled.csv')
# print df_unlabelled.sum(axis=0)

# Load classifiers
classifiers = []
if os.path.exists('../' + folder + '/classifiers.pickle'):
    f = open('../' + folder + '/classifiers.pickle', 'rb')
    classifiers = pickle.load(f)
    f.close()

# # add maximum, voting columns
# prob_columns = [ k for k in df_unlabelled.columns if 'Prob_' in k ]
# pred_columns = [ k for k in df_unlabelled.columns if 'Predicted_' in k ]
# mid = int(len(pred_columns)/2.0)+1
#
# # Approach 1 - Maximum & Approach 2 - Simple Voting
# preds_max = []
# probs_max = []
# preds_vote = []
# probs_vote = []
# # iterate df row by row
# for index, row in df_unlabelled.iterrows():
#     # create df of predictions and probabilities
#     df_temp = pd.DataFrame(index = classifiers, columns=['Predictions', 'Probabilities'])
#     df_temp['Predictions'] = row[pred_columns].values
#     df_temp['Probabilities'] = row[prob_columns].values
#     probs = df_temp['Probabilities'].values
#     preds = df_temp['Predictions'].values
#     ind_max = probs.argmax()
#     # Approach 1 - max pred & max prob as confidence
#     preds_max.append(preds[ind_max])
#     probs_max.append(probs[ind_max])
#
#     # Groupby predicted charge class to aggregate count & mean of probs
#     sums = df_temp.groupby('Predictions')['Probabilities'].sum()
#     # Frequencies of each predicted charge class
#     vc = df_temp['Predictions'].value_counts()
#     # The maximum count
#     vc_max = vc.max()
#     # Another df to hold the counts and mean
#     df_aggr = pd.DataFrame(index=vc.index.values, columns=['counts', 'mean'])
#     df_aggr['counts'] = vc.values
#     df_aggr['mean'] = sums/vc
#     # Filter all the maximum counts
#     df_aggr = df_aggr.loc[df_aggr['counts'] == vc_max]
#     if len(df_aggr) > 1:
#         # index (i.e. charge cls) of row having max. mean
#         pred_vote = df_aggr['mean'].argmax()
#         prob_vote = df_aggr.loc[pred_vote, 'mean']
#     else:
#         # get the index of the mean
#         pred_vote = df_aggr.index.values[0]
#         prob_vote = df_aggr['mean'].values[0]
#     # Approach 2 - simple voting
#     preds_vote.append(pred_vote)
#     probs_vote.append(prob_vote)
#
# df_unlabelled['Maximum_Pred'] = preds_max
# df_unlabelled['Maximum_Prob'] = probs_max
# df_unlabelled['Voting_Pred'] = preds_vote
# df_unlabelled['Voting_Prob'] = probs_vote

# np.append(classifiers, 'Stacking')
# np.append(classifiers, 'Maximum')
np.append(classifiers, 'Voting')
print ', '.join(classifiers)

index = ['Accuracy', 'Precision', 'Recall'] #, 'ROC'
df_accuracies = pd.DataFrame(columns=classifiers, index=index)
actuals = df_unlabelled['Charge Class']
for clf in classifiers:
    vals = []
    preds = df_unlabelled['Predicted_Class_' + clf]
    print 'analyzing : ', clf
    accuracy = round(accuracy_score(actuals, preds)*100, 3)
    precision = round(precision_score(actuals, preds)*100, 3)
    recall = round(recall_score(actuals, preds)*100, 3)
    # print 'ROC = ', roc_curve(actuals, preds)
    vals.append(accuracy)
    vals.append(precision)
    vals.append(recall)
    df_accuracies[clf] = vals

# Voting
clf = 'Voting'
vals = []
preds = df_unlabelled['Voted Output']
print 'analyzing : ', clf
accuracy = round(accuracy_score(actuals, preds)*100, 3)
precision = round(precision_score(actuals, preds)*100, 3)
recall = round(recall_score(actuals, preds)*100, 3)
# print 'ROC = ', roc_curve(actuals, preds)
vals.append(accuracy)
vals.append(precision)
vals.append(recall)
df_accuracies[clf] = vals

# # Maximum
# clf = 'Maximum'
# vals = []
# preds = df_unlabelled['Maximum_Pred']
# print 'analyzing : ', clf
# accuracy = round(accuracy_score(actuals, preds)*100, 3)
# precision = round(precision_score(actuals, preds)*100, 3)
# recall = round(recall_score(actuals, preds)*100, 3)
# # print 'ROC = ', roc_curve(actuals, preds)
# vals.append(accuracy)
# vals.append(precision)
# vals.append(recall)
# df_accuracies[clf] = vals
#
# # Voting
# clf = 'Voting'
# vals = []
# preds = df_unlabelled['Voting_Pred']
# print 'analyzing : ', clf
# accuracy = round(accuracy_score(actuals, preds)*100, 3)
# precision = round(precision_score(actuals, preds)*100, 3)
# recall = round(recall_score(actuals, preds)*100, 3)
# # print 'ROC = ', roc_curve(actuals, preds)
# vals.append(accuracy)
# vals.append(precision)
# vals.append(recall)
# df_accuracies[clf] = vals

# df_unlabelled.to_csv('df_unlabelled.csv')
df_accuracies.to_csv('accuracies_test.csv', index=True, index_label='Classifiers')