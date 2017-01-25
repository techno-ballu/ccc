import os
import pickle
import collections
import pandas as pd
import numpy as np
import logging
import itertools
from scipy.stats.stats import pearsonr
import time
startTime = time.time()
print (time.strftime("%H:%M:%S"))

folder = 'learning_models/models'
# folder = 'classification_models'

# Read unlabelled file
df_features = pd.read_csv('../' + folder + '/debug_features.csv')
print '# of rows = ', str(len(df_features))
print '# of columns = ', str(len(df_features.columns))

# Remove zero variance columns
# calculate the standard deviation in training data
std_dev = df_features.std()
no_original_features = len(std_dev)
std_dev = std_dev.loc[std_dev > 0.03]
selected_features = std_dev.index.values
no_of_dropped_zero = no_original_features - len(selected_features)
print '# Dropped by Zero Variance = ' + str(no_of_dropped_zero)

if len(selected_features) > 0:
    # select variant features
    df_features = df_features[selected_features]
else:
    selected_features = df_features.columns

df_colwise_sums = pd.DataFrame(columns=['sums', 'column'])
sums = df_features.sum(axis=0)
df_colwise_sums['sums'] = sums
df_colwise_sums['column'] = sums.index.values
# print df_colwise_sums

drops = 0
gps = df_colwise_sums.groupby('sums')
for name, group in gps:
    colnames = group['column'].values
    noOfCols = len(colnames)

    if noOfCols <= 1:
        continue

    print name, str(noOfCols)
    print ', '.join(colnames)
    correlations = {}
    columns = colnames.tolist()

    for col_a, col_b in itertools.combinations(columns, 2):
        colA = np.array(df_features.loc[:, col_a].values)
        colB = np.array(df_features.loc[:, col_b].values)
        correlations[col_a + '__' + col_b] = pearsonr(colA, colB)

    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']

    high_correlations = result.loc[result['PCC'].abs() >= 1.0].index.values.tolist()
    high_correlations = set([i.split('__')[0] for i in high_correlations])

    # result = result.loc[result['PCC'].abs() < 1.0]
    print '# Dropped by Pair Wise Correlation =', str(len(high_correlations))
    drops += len(high_correlations)

print '# Dropped by Pair wise correlations = ' + str(drops)
print (time.strftime("%H:%M:%S"))
print 'time taken = ', time.time() - startTime
# df_unlabelled = df_unlabelled.loc[df_unlabelled['occupational_therapy'] == 1]
# print df_unlabelled['Normalized Description']