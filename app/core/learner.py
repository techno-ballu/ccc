import collections
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sknn.mlp import Classifier, Layer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from app.conf import appconfig
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import pickle
import re
from sklearn.cross_validation import StratifiedShuffleSplit
import time
import matplotlib
from matplotlib import pylab, mlab, pyplot as plt
from pylab import *
import logging

# util methods
def saveModel(clf_name, classifier):
    f = open(clf_name + '.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

class Learner(object):
    def __init__(self):
        self.xgb_rounds = appconfig.XGB_ROUNDS
        self.xgb_early_stopping_rounds = appconfig.XGB_EARLY_STOPPING_ROUNDS

    def trainSyntactic(self, clf_name, classifier, x, y, x_test):
        # print classifier
        starttime=time.time()
        classifier.fit(x, y)
        preds, prob_matrix = classifier.predict(x_test)
        logging.debug(' : '+__file__+' : Time taken in training ' + clf_name + ': '+ str(time.time()-starttime))
        print "Time taken in training " + clf_name + ":"
        print time.time()-starttime
        return classifier, preds, prob_matrix

    def trainScikit(self, clf_name, classifier, x, y, x_test):
        # print classifier
        starttime=time.time()
        classifier.fit(x, y)
        prob_matrix = classifier.predict_proba(x_test)
        logging.debug(' : '+__file__+' : Time taken in training ' + clf_name + ': '+ str(time.time()-starttime))
        print "Time taken in training " + clf_name + ":"
        print time.time()-starttime
        return classifier, classifier.predict(x_test), prob_matrix.max(1)

    def cvXGBoost(self, x, y, x_test, y_test, params):

        xg_train = xgb.DMatrix( x, label=y)
        xg_valid = xgb.DMatrix(x_test, label=y_test)

        watchlist = [ (xg_train,'train'), (xg_valid, 'test') ]
        bst = xgb.train(params, xg_train, self.xgb_rounds, watchlist, early_stopping_rounds=self.xgb_early_stopping_rounds, verbose_eval=False )
        pred_probs = bst.predict( xg_valid )
        pred_classes = np.argmax(pred_probs, axis=1)
        return pred_classes

    def trainXGBoost(self, x, y, x_test, params):
        startXGB=time.time()
        xg_x = xgb.DMatrix( x, label=y)
        xg_test = xgb.DMatrix( x_test )
        # bst = xgb.train(params, xg_x, self.xgb_rounds )
        # pred_classes = bst.predict(xg_test).astype(int)
        # params['objective'] = 'multi:softprob'
        bst = xgb.train(params, xg_x, self.xgb_rounds )
        pred_probs = bst.predict(xg_test)
        pred_classes = np.argmax(pred_probs, axis=1)
        logging.debug(' : '+__file__+' : Time taken in training XGB: '+ str(time.time()-startXGB))
        print "Time taken in training XGB:"
        print time.time()-startXGB
        return bst, pred_classes, pred_probs.max(1)

    def trainDeepNN(self, classifier, x, y, x_test):
        startDNN=time.time()
        classifier=Classifier(
                    layers=[
                        Layer("Tanh", units=209), #, pieces=2
                        Layer("Softmax")],
                    learning_rate=0.018,
                    n_iter=25, random_state=786)
        classifier.fit(x, y)
        prob_matrix = classifier.predict_proba(x_test)
        logging.debug(' : '+__file__+' : Time taken in training DeepNN: '+ str(time.time()-startDNN))
        print "time taken in training DeepNN:"
        print time.time()-startDNN
        return classifier, classifier.predict(x_test), prob_matrix.max(1)

    def classify(self, clf_name, classifier, x):
        startClassify = time.time()
        if clf_name == 'XGBoost':
            xg_x = xgb.DMatrix( x )
            pred_probs = classifier.predict(xg_x)
            pred_classes = np.argmax(pred_probs, axis=1)
            pred_probs = pred_probs.max(1)
        elif clf_name in appconfig.syntactic_classifiers:
            pred_classes, pred_probs = classifier.predict(x)
            # pred_probs = classifier.predict_proba(x)
        else:
            pred_classes = classifier.predict(x)
            pred_probs = classifier.predict_proba(x)
            pred_probs = pred_probs.max(1)

        print "Time taken in classifying " + clf_name + ":"
        print time.time()-startClassify
        logging.debug(' : '+__file__+' : Classifying '+clf_name+' :'+str(time.time()-startClassify))
        return pred_classes, pred_probs

class MachineLearning(object):
    def __init__(self, y, no_of_folds = 5, shuffle_split = True):
        # Stratified k-fold
        self.xgb_params = appconfig.XGB_PARAMS
        self.sk_fold = StratifiedKFold(y, no_of_folds, shuffle_split, random_state = 786)
        self.learner = Learner()

    def doCrossValidation(self, clf_name, x_vectors, classifier, y, no_of_classes):
        # Cross Validation
        # print 'Cross validating', clf_name, '...'
        fold = 1

        # The Metrics
        scores = []
        confusion = np.zeros(shape=(no_of_classes, no_of_classes))
        # x_vectors = x_vectors.toarray()

        for train_indices, test_indices in self.sk_fold:
            train_vectors = np.array([x_vectors[i] for i in train_indices])
            train_y = np.array([y[i] for i in train_indices])

            test_vectors = np.array([x_vectors[i] for i in test_indices])
            test_y = np.array([y[i] for i in test_indices])

            if clf_name == 'XGBoost':
                params = self.xgb_params
                params['num_class'] = no_of_classes
                predictions = self.learner.cvXGBoost(train_vectors, train_y, test_vectors, test_y, params)
            elif clf_name=='DeepNN':
                classifier, predictions, probs = self.learner.trainDeepNN(classifier, train_vectors, train_y, test_vectors)
            else:
                print "Time Taken in training "+clf_name+" :"
                classifier, predictions, probs = self.learner.trainScikit(clf_name, classifier, train_vectors, train_y, test_vectors)


            confusion += confusion_matrix(test_y, predictions)
            score = accuracy_score(test_y, predictions)
            scores.append(score)

            # print fold, ' = ', score

            fold += 1

        return predictions, scores, confusion

    def doTraining(self, clf_name, classifier, train_vectors, test_vectors, y, no_of_classes):
        importances = None

        # test on test set
        # if clf_name == 'LDA':
        #     classifier, predictions = self.learner.trainScikit(clf_name, classifier, train_vectors, y, test_vectors)
        if clf_name == 'XGBoost':
            params = self.xgb_params
            params['num_class'] = no_of_classes
            classifier, predictions, probs = self.learner.trainXGBoost(train_vectors, y, test_vectors, params)
        elif clf_name=='DeepNN':
            classifier, predictions, probs = self.learner.trainDeepNN(classifier, train_vectors, y, test_vectors)
            # print classifier.best_params_
        elif clf_name in appconfig.syntactic_classifiers:
            classifier, predictions, probs = self.learner.trainSyntactic(clf_name, classifier, train_vectors, y, test_vectors)
        else:
            classifier, predictions, probs = self.learner.trainScikit(clf_name, classifier, train_vectors, y, test_vectors)

            if clf_name == 'Randomforest':
                importances = classifier.feature_importances_

        return classifier, predictions, probs, importances

class LearningEngine(object):
    def __init__(self, classifiers, train_x, testing_x, y, y_test, df_comparison, features, original_desc_train,
                 original_desc_test, y_ori):
        self.classifiers = classifiers
        self.jaro_train_x = original_desc_train
        self.jaro_test_x = original_desc_test
        self.train_x = train_x
        self.testing_x = testing_x
        self.df_comparison = df_comparison
        self.ml = MachineLearning(y)
        self.features = features
        self.le = LabelEncoder()
        self.y = y
        self.y_test = y_test
        self.y_ori = y_ori
        self.le.fit(y)
        self.dest_folder = appconfig.MODELS_FOLDER_LEARNING
        self.important_words_file = appconfig.IMPORTANT_WORDS_FILE
        self.encoder_file = appconfig.ENCODER_FILE
        self.target = appconfig.TARGET
        self.cvaccuracy=[]

    def crossValidateClassifiers(self):
        startCV=time.time()
        for clf_name in self.classifiers:
            # print '\n', '='*30, clf_name, '='*30
            classifier = self.classifiers[clf_name]

            x = self.train_x
            y = self.y
            if clf_name == 'XGBoost':
                y = self.le.transform(self.y)

            p, s, c = self.ml.doCrossValidation(clf_name, x, classifier, y, len(self.le.classes_))

            if clf_name == 'XGBoost':
                p = self.le.inverse_transform(p)

            # print '\nTotal sentences classified during cross validation:', len(df_training)
            print clf_name, ': cv score:', sum(s)/len(s)
            self.cvaccuracy.append(sum(s)/len(s))
            # print 'Confusion matrix:'
            # print c
        print "Time taken in cross validation:"
        print time.time()-startCV

    def trainClassifiers(self):
        startTrainC=time.time()
        actuals = self.y_test
        important_words = []
        for clf_name in self.classifiers:
            logging.debug(' : '+__file__+' : training classifier : '+ str(clf_name))
            classifier = self.classifiers[clf_name]
            try:
                x = self.train_x
                x_test = self.testing_x
                y = self.y
                if clf_name == 'XGBoost':
                    y = self.le.transform(self.y)

                if clf_name in appconfig.syntactic_classifiers:
                    classifier, predictions, probabilities, importances = self.ml.doTraining(clf_name, classifier, self.jaro_train_x, self.jaro_test_x, self.y_ori, len(self.le.classes_))
                else:
                    classifier, predictions, probabilities, importances = self.ml.doTraining(clf_name, classifier, x, x_test, y, len(self.le.classes_))

                if clf_name == 'XGBoost':
                    predictions = self.le.inverse_transform(predictions)

                if clf_name == 'Randomforest':
                    indices = np.argsort(importances)
                    # indices = indices[100:]
                    important_words = list(reversed( [self.features[i] for i in indices] ))

                    # persist the important words to disk
                    pd.DataFrame(important_words).to_csv('../' + self.dest_folder + '/' + self.important_words_file+'.csv')
                    saveModel('../' + self.dest_folder + '/' + self.important_words_file, important_words)

                # persist the models
                saveModel('../' + self.dest_folder + '/' + clf_name, classifier)

                self.df_comparison.loc[:, 'Predicted_Class_'+clf_name] = predictions
                self.df_comparison.loc[:, 'Prob_'+clf_name] = probabilities
                self.df_comparison.loc[:, 'Correct_'+clf_name] = 0
                self.df_comparison.loc[self.df_comparison['Predicted_Class_'+clf_name] == self.df_comparison[self.target], 'Correct_'+clf_name ] = 1

                # TODO boosting confidences...for Logistic and Levenshtein
                if clf_name == 'Jaro':
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.8) & (self.df_comparison['Prob_'+clf_name] < 0.85), 'Prob_'+clf_name] = 0.85
                elif clf_name == 'Logistic_OvR':
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.8) & (self.df_comparison['Prob_'+clf_name] < 0.9), 'Prob_'+clf_name] = 0.96
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.7) & (self.df_comparison['Prob_'+clf_name] < 0.8), 'Prob_'+clf_name] = 0.92
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.6) & (self.df_comparison['Prob_'+clf_name] < 0.7), 'Prob_'+clf_name] = 0.76
                elif clf_name == 'DLevenshtein':
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.8) & (self.df_comparison['Prob_'+clf_name] < 0.9), 'Prob_'+clf_name] = 0.96
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.7) & (self.df_comparison['Prob_'+clf_name] < 0.8), 'Prob_'+clf_name] = 0.93
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.6) & (self.df_comparison['Prob_'+clf_name] < 0.7), 'Prob_'+clf_name] = 0.88
                    self.df_comparison.loc[(self.df_comparison['Prob_'+clf_name] >= 0.5) & (self.df_comparison['Prob_'+clf_name] < 0.6), 'Prob_'+clf_name] = 0.81

                # print '\n', '='*20, 'metrics on test set', '='*20
                # print clf_name, ': accuracy predictions on test set = ', accuracy_score(actuals, predictions)
                # print 'Confusion Matrix:\n', confusion_matrix(actuals, predictions)
                # print 'Classification Report:\n', classification_report(actuals, predictions)

                # if clf_name == 'XGBoost':
                #     xg_unlabelled = xgb.DMatrix( unlabelled_vectors )
                #     predictions_encoded = bst.predict(xg_unlabelled).astype(int)
                #     predictions = le.inverse_transform(predictions_encoded)
                # else:
                #     predictions = classifier.predict(unlabelled_vectors)
                #
                # df_unlabelled['Predicted_Class_'+clf_name] = predictions
            except Exception as e:
                logging.error(e)
                logging.debug(' : '+__file__+' : Classifier '+ str(clf_name) +' was not trained successfully.')

        # persist the models
        saveModel('../' + self.dest_folder + '/' + self.encoder_file, self.le)
        print "Total time taken by TrainClassifiers:"
        print time.time()-startTrainC
        return self.df_comparison, important_words

class FeatureImportance(object):
    def __init__(self, important_words):
        self.dest_folder = appconfig.MODELS_FOLDER_LEARNING
        self.data_folder = appconfig.DATA_MODELS_LEARNING
        self.data_folder_classification = appconfig.MODELS_FOLDER_CLASSIFICATION
        self.origDesc = appconfig.ORIGINAL_DESCRIPTION
        self.important_words = important_words
        self.abbreviations_reverse_mapping = pd.read_csv('../' + self.data_folder_classification + '/' + appconfig.ABBREVIATIONS_FILE, index_col='Expansion')
        self.abbreviations = list(self.abbreviations_reverse_mapping.index.values)

    def extractFeatures(self, df, column):

        cleaned_claims_test = np.array( df[column].values )
        original_descriptions = np.array( df[self.origDesc].values )
        features = []

        for x, claim in enumerate(cleaned_claims_test):
            featurecounter=1
            claim_features = []

            original_description = original_descriptions[x]
            claim_feats1=[]
            claim_feats = claim.split()
            claim_feats=[i for i in claim_feats if (i.isdigit()==False)]
            if len(claim_feats) > 4:
                claim_feats1 = [i for i in claim_feats if (i in self.important_words)]
            if len(claim_feats1)!=0:
                claim_feats=claim_feats1

            claim_feats=list(set(claim_feats))

            for i, feat in enumerate(claim_feats):
                # if i < 4:
                if bool(re.search(r'\d', claim_feats[i]))==False:
                    claim_feats_splitted=claim_feats[i].split('/')
                else:
                    claim_feats_splitted=[claim_feats[i]]

                for word in claim_feats_splitted:
                    expanded = word

                    if word in self.abbreviations:
                        abbr_exps = self.abbreviations_reverse_mapping.loc[word].values
                        for abbr in abbr_exps:
                            if len(abbr)==1:
                                abbr = abbr[0]
                            abbr = str(abbr).lower()
                            if abbr in original_description.lower().split(' '):
                                expanded = abbr
                                break

                    # df.loc[index, 'Feature ' + str(featurecounter)] = expanded
                    claim_features.append(expanded)
                    featurecounter+=1

                    if featurecounter>4:
                        break
                if featurecounter>4:
                    break

            claim_features += [''] * (4 - len(claim_features))
            features.append(claim_features)

        features = np.array(features)
        for i in range(4):
            df['Feature ' + str(i+1)] = features[:,i]
            df['Feature ' + str(i+1)] = df['Feature ' + str(i+1)].apply(lambda x: x[:50])

        if appconfig.TEXT_FEATURE in df.columns:
            df[appconfig.TEXT_FEATURE] = df[appconfig.TEXT_FEATURE].apply(lambda x: x[:255])
        if 'CHNGS' in df.columns:
            df['CHNGS'] = df['CHNGS'].apply(lambda x: x[:400])
        return df

class Ensembler(object):
    def __init__(self, y, df_predictions, classifiers):
        self.y = y
        self.le = LabelEncoder()
        self.le.fit(y)
        self.df_predictions = df_predictions.copy()
        self.classifiers = classifiers
        self.dest_folder = appconfig.MODELS_FOLDER_LEARNING
        self.target = appconfig.TARGET

    def encodePredictions(self):
        self.names = []
        for clf_name in self.classifiers:
            col_name = 'Predicted_Class_'+clf_name
            self.df_predictions[col_name] = self.le.transform(self.df_predictions['Predicted_Class_'+clf_name].values)
            self.names.append(col_name)
            # self.names.append('Prob_'+clf_name)

        # self.names.append(self.target)

    def removeDisambiguousCases(self):
        self.df_predictions['Sum Correct'] = 0
        for clf_name in self.classifiers:
            self.df_predictions['Sum Correct'] += self.df_predictions['Correct_'+clf_name]

        print 'Removing disambiguous cases = ', len(self.df_predictions.loc[self.df_predictions['Sum Correct'] == 0])
        self.df_predictions.drop('Sum Correct', axis=1, inplace=True)

    def crossValidateEnsembleFit(self):
        # spilt into stratified training & test sets
        print pd.DataFrame(self.y)[0].value_counts()
        sss = StratifiedShuffleSplit(self.y, 5, 0.25, random_state=786)

        # train_index, test_index = list(sss)[0]
        scores = []
        fold = 1
        for train_index, test_index in sss:
            df_ensemble_train = self.df_predictions.iloc[train_index]
            df_ensemble_test = self.df_predictions.iloc[test_index]

            ensemble_train_x = df_ensemble_train[[k for k in self.names]].values
            ensemble_train_y = df_ensemble_train[self.target].values

            ensemble_test_x = df_ensemble_test[[k for k in self.names]].values
            ensemble_test_y = df_ensemble_test[self.target].values

            # print ensemble_train_x.shape, ensemble_train_y.shape, ensemble_test_x.shape, ensemble_test_y.shape

            self.ensembler = RandomForestClassifier(random_state=12246)
            # self.ensembler = DecisionTreeClassifier(random_state=98876)
            # self.ensembler = LogisticRegression(multi_class='ovr')
            self.ensembler.fit(ensemble_train_x, ensemble_train_y)
            voted_output = self.ensembler.predict(ensemble_test_x)
            score = accuracy_score(ensemble_test_y, voted_output)
            scores.append(score)
            # print 'Ensemble', fold, ' = ', score
            fold += 1

        ensemble_x = self.df_predictions[[k for k in self.names]].values
        ensemble_y = self.df_predictions[self.target].values
        self.ensembler = RandomForestClassifier(random_state=12246)
        self.ensembler.fit(ensemble_x, ensemble_y)
        print 'Ensemble accuracy = ', np.mean(scores)

    def fitEnsembler(self):
        ensemble_x = self.df_predictions[[k for k in self.names]].values
        ensemble_y = self.df_predictions[self.target].values
        self.ensembler = RandomForestClassifier(random_state=12246)
        self.ensembler.fit(ensemble_x, ensemble_y)

    def ensemblePredictions(self):
        ensemble_x = self.df_predictions[[k for k in self.names]].values
        voted_output = self.ensembler.predict(ensemble_x)
        self.df_predictions['Voted Output'] = voted_output

        for clf_name in self.classifiers:
            col_name = 'Predicted_Class_'+clf_name
            self.df_predictions[col_name] = self.le.inverse_transform(self.df_predictions[col_name].values)

        return self.df_predictions

    def pickleObjects(self, clf_name, encoder):
        # persist the models
        saveModel('../' + self.dest_folder + '/' + clf_name, self.ensembler)
        saveModel('../' + self.dest_folder + '/' + encoder, self.le)

class Voter(object):
    def __init__(self, df_predictions, classifiers):
        self.df_predictions = df_predictions.copy()
        self.classifiers = classifiers
        self.target = appconfig.TARGET
        self.prob_columns = [ k for k in df_predictions.columns if 'Prob_' in k ]
        self.pred_columns = [ k for k in df_predictions.columns if 'Predicted_' in k ]

    def votingPredictions(self):
        logging.info(' : '+__file__+' : In votingPredictions() : to vote ' + str(len(self.classifiers.keys())) )
        # initialize lists to hold voted charge class & probability
        preds_vote = []
        voted_algos = []

        # Adding empty checks, both predicted & probability columns should exist.
        if len(self.prob_columns) == 0 or len(self.pred_columns) == 0:
            logging.info(' : '+__file__+' : no predicted/probability columns detected!')
            return self.df_predictions

        # iterate df row by row
        for index, row in self.df_predictions.iterrows():
            orig_desc = ''
            if appconfig.ORIGINAL_DESCRIPTION in row:
                orig_desc = row[appconfig.ORIGINAL_DESCRIPTION]

            # create temp df of predictions and probabilities
            df_temp = pd.DataFrame(index = self.classifiers.keys(), columns=['Predictions', 'Probabilities'])
            df_temp['Predictions'] = row[self.pred_columns].values
            df_temp['Probabilities'] = row[self.prob_columns].values

            # Groupby predicted charge class to aggregate count & mean of probs
            sums = df_temp.groupby('Predictions')['Probabilities'].sum()
            # Frequencies of each predicted charge class
            vc = df_temp['Predictions'].value_counts()
            # The maximum count
            vc_max = vc.max()
            # Another temp df to hold the counts and mean
            df_aggr = pd.DataFrame(index=vc.index.values, columns=['counts', 'mean'])
            df_aggr['counts'] = vc.values
            df_aggr['mean'] = sums/vc
            # Filter all the maximum counts
            df_aggr = df_aggr.loc[df_aggr['counts'] == vc_max]
            pred_vote = ''
            voted_algo = ''

            if len(df_aggr) > 1:
                # index (i.e. charge cls) of row having max. mean
                pred_vote = df_aggr['mean'].argmax()
                df_temp = df_temp.loc[df_temp['Predictions'] == pred_vote]
                voted_algo = ', '.join(df_temp.index.values)
            elif len(df_aggr) == 1:
                # get the index of the mean
                pred_vote = df_aggr.index.values[0]
                df_temp = df_temp.loc[df_temp['Predictions'] == pred_vote]
                voted_algo = ', '.join(df_temp.index.values)

            # Approach 2 - simple voting
            preds_vote.append(pred_vote)
            voted_algos.append(voted_algo)
            logging.debug(' : '+__file__+' : '+orig_desc + '=' + pred_vote)

        self.df_predictions['Voted Output'] = preds_vote
        self.df_predictions['Voting Algorithm'] = voted_algos

        for clf_name in self.classifiers:
            col_name = 'Predicted_Class_'+clf_name

            self.df_predictions['Correct_'+clf_name] = 0
            self.df_predictions.loc[ self.df_predictions[col_name] == self.df_predictions['Voted Output'], 'Correct_'+clf_name ] = 1

        return self.df_predictions

    def maximumOfPredictions(self):
        # initialize lists to hold maximum charge class & probability
        preds_max = []
        # probs_max = []

        # iterate df row by row
        for index, row in self.df_predictions.iterrows():
            # get probabilities and predictions as arrays
            probs = row[self.prob_columns].values
            preds = row[self.pred_columns].values
            # argmax of max probability
            ind_max = probs.argmax()
            # Approach 1 - max pred & max prob as confidence
            preds_max.append(preds[ind_max])
            # probs_max.append(probs[ind_max])

        self.df_predictions['Voted Output'] = preds_max
        # self.df_predictions[appconfig.CONF_SCORE] = probs_max

        return self.df_predictions

class LearningStats(object):
    def __init__(self, classifiers, classes):
        self.classifiers = classifiers.keys()
        self.classifiers.append('Voted Output')
        # self.predictions = collections.OrderedDict()
        classes.sort()
        self.classes = classes
        index = []
        index.extend(self.classes)
        # # index.append('Voted Output')
        # columns = []
        # columns.append('# of claim lines')
        # columns.extend(classifiers.keys())
        # columns.append('Voted Output')
        self.precision = pd.DataFrame   (index=index) # , columns=columns
        self.recall = pd.DataFrame(index=index) # , columns=columns
        # self.accuracy = pd.DataFrame(index=['Overall Accuracy']) #, columns=columns
        self.target = appconfig.TARGET
        self.dest_folder = appconfig.MODELS_FOLDER_LEARNING
        # self.plots_folder = appconfig.PLOTS_FOLDER
        # self.combination = combination

    def calculateStats(self, df_predictions, prefix = ''):
        actuals = df_predictions[self.target]

        counts = df_predictions['Charge Class'].value_counts()
        self.precision['# of claim lines'] = counts
        self.recall['# of claim lines'] = counts
        # print counts

        accuracies = []
        for clf_name in self.classifiers:
            if clf_name == 'Voted Output':
                p = df_predictions['Voted Output']
            else:
                p = df_predictions['Predicted_Class_'+clf_name]

            precision, recall, fscore, support = precision_recall_fscore_support(actuals, p)
            accuracy = accuracy_score(actuals, p)
            self.precision['Predicted_Class_'+clf_name] = precision
            self.recall['Predicted_Class_'+clf_name] = recall
            # self.accuracy['Predicted_Class_'+clf_name] = accuracy
            accuracies.append(accuracy)
            print clf_name, ': accuracy predictions on test set = ', accuracy

        self.precision.to_csv('../' + self.dest_folder + '/' + prefix + '_precision.csv', index=True, index_label='Charge Class')
        self.recall.to_csv('../' + self.dest_folder + '/' + prefix + '_recall.csv', index=True, index_label='Charge Class')
        # self.accuracy.to_csv('../' + self.dest_folder + '/' + prefix + '_accuracy.csv', index=True)

        return accuracies, self.classifiers





class ConfidenceCalculation(object):
    def __init__(self, classifiers, df_predictions):
        self.classifiers=classifiers
        self.df_predictions=df_predictions
        self.confidence_col = appconfig.CONF_SCORE
        self.correct_columns=np.array(['Correct_'+i for i in self.classifiers])
        self.prob_columns=np.array(['Prob_'+i for  i in self.classifiers])

    def calculate(self):
        logging.debug(' : '+__file__+' : Confidence calculation by calculate()')
        correct_matrix=np.array(self.df_predictions[self.correct_columns])
        prob_matrix=np.array(self.df_predictions[self.prob_columns])
        conf_score=np.mean(correct_matrix*prob_matrix,axis=1)
        self.df_predictions[self.confidence_col]=conf_score
        return self.df_predictions

    def calculateByBoosting(self):
        logging.debug(' : '+__file__+' : Confidence calculation by calculateByBoosting()')

        conf_scores = []
        for index, row in self.df_predictions.iterrows():
            corrects = np.array(row[self.correct_columns].values)
            probs = np.array(row[self.prob_columns].values)
            sum_corrects = np.sum(corrects)
            if sum_corrects == 3:
                conf_scores.append(np.sum(corrects*probs)/3)
            else:
                conf_scores.append(np.sum(corrects*probs)/4)

        self.df_predictions[self.confidence_col]=conf_scores
        return self.df_predictions

