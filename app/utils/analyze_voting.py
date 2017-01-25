import pandas as pd

df_unlabelled = pd.read_csv('/home/abzooba/git/equian_ml_ensemble/app/comparison/algorithms trials/6324/5. Logistic + Deep_NN + Jaro/df_unlabelled.csv')
# print df_unlabelled.columns

actuals = df_unlabelled['CHRG_CLS']
jaro = df_unlabelled['Predicted_Class_Jaro']
logistic = df_unlabelled['Predicted_Class_Logistic_OvR']
deepnn = df_unlabelled['Predicted_Class_DeepNN']
voted = df_unlabelled['Voted Output']

# add algo column
pred_columns = [ k for k in df_unlabelled.columns if 'Predicted_' in k ]

# iterate df row by row
voted_algos = []
for index, row in df_unlabelled.iterrows():
    voted_output = row['Voted Output']

    # create temp df of predictions and probabilities
    df_temp = pd.DataFrame(index = ['Jaro', 'Logistic', 'DeepNN'], columns=['Predictions'])
    df_temp['Predictions'] = row[pred_columns].values

    df_temp = df_temp.loc[df_temp['Predictions'] == voted_output]
    voted_algo = ', '.join(df_temp.index.values)
    voted_algos.append(voted_algo)

df_unlabelled['Voting Algorithm'] = voted_algos

# Add match column
df_unlabelled['Match'] = 0
df_unlabelled.loc[actuals == voted, 'Match'] = 1
# print df_unlabelled['Match'].value_counts()

# Add a new column to analyse
df_unlabelled['kind of voting'] = 'NA'
df_unlabelled.loc[(jaro == logistic) & (logistic == deepnn), 'kind of voting'] = 'all agree'
df_unlabelled.loc[(jaro != logistic) & (jaro != deepnn), 'kind of voting'] = 'all disagree'
df_unlabelled.loc[(logistic == jaro) & (logistic != deepnn), 'kind of voting'] = 'deepnn disagree'
df_unlabelled.loc[(logistic == deepnn) & (logistic != jaro), 'kind of voting'] = 'jaro disagree'
df_unlabelled.loc[(deepnn == jaro) & (logistic != deepnn), 'kind of voting'] = 'logistic disagree'

print df_unlabelled['kind of voting'].value_counts()
types = df_unlabelled['kind of voting'].unique()

groups = df_unlabelled.groupby('kind of voting')
group_names = []
totals = []
corrects = []
wrongs = []
for name, group in groups:
    group_names.append(name)
    accurate = group['Match'].sum()
    total = len(group)
    misclassified = total - accurate
    corrects.append(accurate)
    wrongs.append(misclassified)
    totals.append(total)

    # if name == 'all disagree':
    #     correct = group.loc[group['Match'] == 1]
    #     wrong = group.loc[group['Match'] == 0]
    #     # print 'All disagree:'
    #     print 'Correct:'
    #     print correct['Voting Algorithm'].value_counts()
    #     print 'Wrong:'
    #     print wrong['Voting Algorithm'].value_counts()

df_results = pd.DataFrame(index=group_names, columns=['Total', 'Accurate', 'Misclassified'])
df_results['Total'] = totals
df_results['Accurate'] = corrects
df_results['Misclassified'] = wrongs

print df_results
df_results.to_csv('kind_of_voting.csv', index=True, index_label='kind of voting')
df_unlabelled.to_csv('df_unlabelled.csv', index=False)