from app.core import data_preprocessor
from app.core import jaro_wrinkler_algo
import logging
import pandas as pd
import numpy as np

set2Final = pd.read_csv('../release10/descrip_10k.csv')
trainingData = pd.read_csv('../release10/training_6960.csv')
print set2Final.columns
print trainingData.columns
# extract the columns
classification_sents = np.array( set2Final['Original Description'].values )
training_sents = np.array( trainingData['Original Description'].values )
training_chargeclasses = np.array( trainingData['Charge Class'].values )
columns = ['Original Description', 'Charge Class']
df_complete = pd.DataFrame(columns = columns)

# initialize containers
classes = []
scores = []
matches = []
counts = []

# iterate sentences in classification set
for i, new_description in enumerate(classification_sents):
    print i, new_description
    charge_classes = []
    distances=[]
    match = []

    # iterate sentences in training set
    for j, training_sent in enumerate(training_sents):
        charge_class = training_chargeclasses[j]
        distance = jaro_wrinkler_algo.JaroWinklerDistance(new_description, training_sent)

        if distance >= 0.80:
            charge_classes.append(charge_class)
            distances.append(distance)
            match.append(training_sent)

    charge_classes = list(set(charge_classes))
    counts.append(len(match))
    if len(match) > 0:
        matches.append( ', '.join(match) )
        classes.append( ', '.join(charge_classes) )
        if len(charge_classes)==1:
            scores.append(sum(distances)/len(distances))
        else:
            scores.append('')
    else:
        matches.append('')
        classes.append('')
        scores.append('')


df_complete['Original Description'] = set2Final['Original Description']
df_complete['Charge Class'] = classes
df_complete['Confidence Score'] = scores
df_complete['Matches']=matches
df_complete['Counts']=counts

df_complete.to_csv('../release10/Matches_count.csv',index=False)
# set2Final=pd.DataFrame()
# set2similars=pd.DataFrame()
# set2Final=df_complete[df_complete['Flag']!=1]
# set2Final=set2Final[['Original Description','Normalized Description','Changes']]
# logging.info(' : '+__file__+' : len of set2Final = ' + str(len(set2Final)))
# set2similars=df_complete[df_complete['Flag']==1]
# logging.info(' : '+__file__+' : len of set2similars = ' + str(len(set2similars)))