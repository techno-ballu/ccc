import pandas as pd

df_actuals = pd.read_csv('../../data/testing_1330.csv')
df_predicted = pd.read_csv('../../data/df_unlabelled_case_g_5k.csv')
df_classified = pd.read_csv('../../data/clm_ln_clsf_rep_out_case_g_5k.csv')
print 'actuals columns:'
print df_actuals.columns
print 'predicted columns:'
print df_predicted.columns
print 'classified columns:'
print df_classified.columns

# clean duplicates
df_predicted.drop_duplicates('Original Description', inplace=True)
print 'length of actuals = ', str(len(df_actuals))
print 'length of predicted = ', str(len(df_predicted))
print 'length of classified = ', str(len(df_classified))

# handle mismatches due to cleaning
df_actuals.loc[df_actuals['ORIG_DESC'] == 'SET FEED EXT 60"', 'ORIG_DESC'] = 'SET FEED EXT 60'
df_actuals.loc[df_actuals['ORIG_DESC'] == 'GLUCOSE BLOOD TEST - EYER ', 'ORIG_DESC'] = 'GLUCOSE BLOOD TEST - EYER'

df_joined = pd.merge(df_predicted, df_actuals, how='inner', left_on='Original Description', right_on='ORIG_DESC',
                     sort=True, suffixes=('_pred', '_actual'), copy=True, indicator=False)
print df_joined.shape
print  df_joined.columns

# Match columns in Predicted
actuals = df_joined['CHRG_CLS']
jaro = df_joined['Predicted_Class_Jaro']
leven = df_joined['Predicted_Class_DLevenshtein']
logistic = df_joined['Predicted_Class_Logistic_OvR']
deepnn = df_joined['Predicted_Class_DeepNN']
voted = df_joined['Voted Output']

df_joined['Match_Jaro'] = 0
df_joined.loc[actuals == jaro, 'Match_Jaro'] = 1

df_joined['Match_DLevenshtein'] = 0
df_joined.loc[actuals == leven, 'Match_DLevenshtein'] = 1

df_joined['Match_Logistic'] = 0
df_joined.loc[actuals == logistic, 'Match_Logistic'] = 1

df_joined['Match_DeepNN'] = 0
df_joined.loc[actuals == deepnn, 'Match_DeepNN'] = 1

df_joined['Match_Voted'] = 0
df_joined.loc[actuals == voted, 'Match_Voted'] = 1

# # Another join for disjoint set
# rem_descs = df_classified['ORIG_DESC'].values
# pred_descs = df_joined['Original Description'].values
#
# not_found = [d for d in rem_descs if d not in pred_descs]
# df_classified = df_classified[df_classified['ORIG_DESC'].isin(not_found)]
# print len(df_classified)

df_joined2 = pd.merge(df_classified, df_actuals, how='inner', on='ORIG_DESC',
                     sort=True, suffixes=('_pred', '_actual'), copy=True, indicator=False)
print df_joined2.shape
print  df_joined2.columns

# Match columns in Classified
actuals = df_joined2['CHRG_CLS_actual']
voted = df_joined2['CHRG_CLS_pred']

df_joined2['Match_Voted'] = 0
df_joined2.loc[actuals == voted, 'Match_Voted'] = 1

df_joined = df_joined[['Original Description', 'Normalized Description', 'Predicted_Class_Jaro', 'Prob_Jaro',
                       'Predicted_Class_DLevenshtein', 'Prob_DLevenshtein', 'Predicted_Class_Logistic_OvR',
                       'Prob_Logistic_OvR', 'Predicted_Class_DeepNN', 'Prob_DeepNN', 'Voted Output', 'CHRG_CLS',
                       'Voting Algorithm', 'Confidence Score', 'Match_Jaro', 'Match_DLevenshtein', 'Match_Logistic',
                       'Match_DeepNN', 'Match_Voted']]
df_joined2 = df_joined2[['ORIG_DESC', 'NORM_DESC_pred', 'CHRG_CLS_actual', 'CHRG_CLS_pred',
                       'CONF_SCORE', 'Match_Voted']]
df_joined.to_csv('../../data/df_predicted.csv', index=False)
df_joined2.to_csv('../../data/df_jaro.csv', index=False)