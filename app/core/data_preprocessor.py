from __future__ import division

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
from app.conf import appconfig
from app.core import jaro_wrinkler_algo
import jellyfish as jf
import logging
# from app.core import learner

class DataPreprocessor(object):
    def __init__(self, target, data_folder, destination_folder, training_file=None, test_file=None):
        self.target = target
        self.delimit=appconfig.DELIMITER
        self.data_folder = data_folder
        self.training_file = False
        self.character_encoding=appconfig.ENCODING_TO_USE
        self.test_file = False
        self.important_words_file=appconfig.IMPORTANT_WORDS_FILE
        # self.unlabelled_file = False
        if training_file:
            self.training_file = training_file
        if test_file:
            self.test_file = test_file
        # if unlabelled_file:
        #     self.unlabelled_file = unlabelled_file
        self.text_feature=appconfig.TEXT_FEATURE
        self.destination_folder = destination_folder
        # self.reason_code_col = reason_code_col

    def loadData(self):
        if self.training_file:
            self.df_training = pd.read_csv('../' + self.data_folder + '/' + self.training_file,encoding=self.character_encoding)
            print '# of training samples=', len(self.df_training)
            # print self.df_training[self.target].value_counts()
        if self.test_file:
            self.df_testing = pd.read_csv('../' + self.data_folder + '/' + self.test_file,encoding=self.character_encoding)
            print '# of test samples=', len(self.df_testing)
            # print self.df_testing[self.target].value_counts()



    def splitTestTrain(self):
        self.df_training,self.df_testing= self.stratifiedSplitData(self.df_training,0.1,self.target)

    def stratifiedSplitData(self, df, test_size, column):
        y = df[column].values

        # spilt into stratified training & test sets
        sss = StratifiedShuffleSplit(y, 1, test_size = int(test_size), random_state=786)
        train_index, test_index = list(sss)[0]

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]


        return df_train, df_test

    def shuffleSplitData(self, n, split_pct, df, random_state):

        # spilt into training & test sets
        ss = ShuffleSplit(n, n_iter=1, test_size=split_pct, random_state=random_state)
        train_index, test_index = list(ss)[0]

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        return df_train, df_test

    def saveData(self):

        data = dict()

        # save the training & test sets
        if self.training_file:
            self.df_training.to_csv('../' + self.destination_folder + '/training.csv', index=False,encoding=self.character_encoding)
            data['training'] =  self.df_training

        if self.test_file:
            self.df_testing.to_csv('../' + self.destination_folder + '/testing.csv', index=False,encoding=self.character_encoding)
            data['testing'] =  self.df_testing


        # print '\nSaved', ', '.join(data.keys()), 'data in', self.destination_folder, 'folder'
        return data

    def upsampleMinorChargeClasses(self, min_count):
        data = dict()

        if self.training_file:
            data['training'] = self.df_training

        if self.test_file:
            data['testing'] = self.df_testing

        chrgClass_valueCounts = self.df_training[self.target].value_counts()


        if min(chrgClass_valueCounts) < min_count:
            vc = chrgClass_valueCounts[chrgClass_valueCounts < min_count]
            for i in range(0,len(vc)):
                chrgClass = vc.index[i]
                count = vc[chrgClass]
                df_chrgClass = self.df_training[self.df_training[self.target]==chrgClass]
                n = int(min_count/count)
                if n > 1:
                   df_chrgClass = df_chrgClass.append([df_chrgClass]*(n-1),ignore_index=True)
                df_chrgClass = df_chrgClass.ix[np.random.choice(df_chrgClass.index, min_count-count)]
                self.df_training = self.df_training.append(df_chrgClass,ignore_index=True)
                data['training'] =  self.df_training
            return data

        return data

    def setSimilarToTraining(self,set2Complete,trainingData):
        classification_sents = np.array( set2Complete['Original Description'].values )
        training_sents = np.array( trainingData['Original Description'].values )
        training_chargeclasses = np.array( trainingData['Charge Class'].values )
        columns = ['Original Description', 'Normalized Description', 'Charge Class', 'Changes', 'Confidence Score','Flag']
        df_complete = pd.DataFrame(columns = columns)

        classes = []
        conf_score=[]
        flags=[]
        matches=[]

        for i, ref_sent in enumerate(classification_sents):
            similar = []
            charge_classes = []
            distances=[]

            for j, training_sent in enumerate(training_sents):
                charge_class = training_chargeclasses[j]
                distance = jaro_wrinkler_algo.JaroWinklerDistance(ref_sent, training_sent)


                if distance >= appconfig.JARO_WINKLER_THRESHOLD:
                    similar.append(training_sent)
                    distances.append(distance)
                    charge_classes.append(charge_class)

            charge_classes = list(set(charge_classes))

            if len(similar) > 0:
                matches.append( ', '.join(similar) )
                classes.append( ', '.join(charge_classes) )
                if len(charge_classes)==1:
                    conf_score.append(sum(distances)/len(distances))
                    flags.append(1)
                else:
                    flags.append('')
                    conf_score.append('')
            else:
                matches.append('')
                classes.append('')
                flags.append('')
                conf_score.append('')

        df_complete['Original Description'] = set2Complete['Original Description']
        df_complete['Normalized Description'] = set2Complete['Normalized Description']
        df_complete['Charge Class'] = classes
        df_complete['Changes'] = set2Complete['Changes']
        df_complete['Confidence Score'] = conf_score
        df_complete['Flag']=flags
        df_complete['Matches']=matches

        # df_complete.to_csv('../'+self.data_folder+'/Matches_bugFixing.csv',index=False)
        set2Final=pd.DataFrame()
        set2similars=pd.DataFrame()
        set2Final=df_complete[df_complete['Flag']!=1]
        set2Final=set2Final[['Original Description','Normalized Description','Changes']]
        set2similars=df_complete[df_complete['Flag']==1]
        return set2Final, set2similars

    def setSimilarToTrainingNew(self, set2Complete, trainingData):
        logging.info(' : '+__file__+' : '+'In setSimilarToTrainingNew() method.')
        # extract the columns
        classification_sents = np.array( set2Complete['Original Description'].values )
        training_sents = np.array( trainingData['Original Description'].values )
        training_chargeclasses = np.array( trainingData['Charge Class'].values )
        columns = ['Original Description', 'Normalized Description', 'Charge Class', 'Changes', 'Confidence Score','Flag']
        df_complete = pd.DataFrame(columns = columns)

        # initialize containers
        flags=[]
        classes = []
        scores = []
        matches = []

        # iterate sentences in classification set
        for i, new_description in enumerate(classification_sents):
            charge_classes = []
            distances=[]
            match = []

            # iterate sentences in training set
            for j, training_sent in enumerate(training_sents):
                charge_class = training_chargeclasses[j]
                distance = jaro_wrinkler_algo.JaroWinklerDistance(new_description, training_sent)

                charge_classes.append(charge_class)
                distances.append(distance)
                match.append(training_sent)

            # find index of max distance
            index_max = np.argmax(distances)

            # The distance of closest match
            max_dist = distances[index_max]

            if max_dist >= appconfig.JARO_WINKLER_THRESHOLD:
                # distance exceeds Jaro threshold
                classes.append(charge_classes[index_max])
                scores.append(max_dist)
                flags.append(1)
                matches.append(match[index_max])
                # logging.debug(' : '+__file__+' : '+ str(new_description) + ':' + match[index_max] + '=' + str(max_dist))
            else:
                # distance does not exceed Jaro threshold
                classes.append('')
                scores.append('')
                flags.append('')
                matches.append('')

        df_complete['Original Description'] = set2Complete['Original Description']
        df_complete['Normalized Description'] = set2Complete['Normalized Description']
        df_complete['Charge Class'] = classes
        df_complete['Changes'] = set2Complete['Changes']
        df_complete['Confidence Score'] = scores
        df_complete['Flag']=flags
        df_complete['Matches']=matches

        # df_complete.to_csv('../classification_models/Matches_bugFixing.csv',index=False)
        set2Final=pd.DataFrame()
        set2similars=pd.DataFrame()
        set2Final=df_complete[df_complete['Flag']!=1]
        set2Final=set2Final[['Original Description','Normalized Description','Changes']]
        logging.info(' : '+__file__+' : len of set2Final = ' + str(len(set2Final)))
        set2similars=df_complete[df_complete['Flag']==1]
        # TODO boosting Jaro conf...
        set2similars.loc[(set2similars['Confidence Score'] >= 0.85) & (set2similars['Confidence Score'] < 0.9), 'Confidence Score'] = 0.91
        logging.info(' : '+__file__+' : len of set2similars = ' + str(len(set2similars)))
        return set2Final, set2similars

