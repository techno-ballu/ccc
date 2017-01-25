import jellyfish as jf
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import logging
import re

ORIGINAL_DESCRIPTION = 'Original Description'
TARGET = 'Charge Class'

# This method calculates the Jaro Wrinkler distance between two sentences
def JaroWinklerDistance(sent1, sent2):
    d=0.0

    # adding Try-Catch
    try:
        sent1 = unicode(sent1, encoding='iso-8859-1')
        sent2 = unicode(sent2, encoding='iso-8859-1')

        # Remove all white spaces and all special characters from the string for Jaro
        sent1 = re.sub('[^A-Za-z0-9+%.>/&]',' ',sent1)
        sent2 = re.sub('[^A-Za-z0-9+%.>/&]',' ',sent2)
        sent1 = ''.join(sent1.split())
        sent2 = ''.join(sent2.split())
        d = jf.jaro_distance(sent1, sent2)
        # logging.debug(' : '+__file__+' : : distance='+str(round(d, 3)) + ' : ' + sent1 + ' : ' + sent2)
    except:
        logging.warning(' : '+__file__+' : : exception occured in JaroWinklerDistance for ' + sent1 + ' : ' + sent2)
        d = 0.0

    return round(d, 2)


class JaroWrinklerClassifier(BaseEstimator, ClassifierMixin):
    """ Jaro Wrinkler classifier
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self):
        self.df_training = None
        self.predictions = []
        self.probabilities = []

    def fit(self, descriptions = None, charge_classes = None):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        df_training : pandas dataframe, shape = [n_samples, 2]
            The training input samples containing original descriptions and charge classes.
        Returns
        -------
        self : object
            Returns self.
        """
        if descriptions is not None and charge_classes is not None and len(descriptions) > 0 and len(charge_classes) > 0:
            if len(descriptions) == len(charge_classes):
                logging.info(' : '+__file__+' : : fit Jaro Wrinkler Classifier...')
                self.df_training = pd.DataFrame(columns=[ORIGINAL_DESCRIPTION, TARGET])
                self.df_training[ORIGINAL_DESCRIPTION] = descriptions
                self.df_training[TARGET] = charge_classes
            else:
                logging.info(' : '+__file__+' : : Jaro Classifier not fitted, no of descriptions != no of charge classes.')

        # Return the classifier
        return self

    def __computeJaros__(self, new_descriptions):
        # Return corner cases
        # Check is fit had been called
        if self.df_training is None:
            logging.debug(' : '+__file__+' : : df_training is None...')
            return

        if ORIGINAL_DESCRIPTION not in self.df_training:
            logging.debug(' : '+__file__+' : : ORIGINAL_DESCRIPTION column not in df_training...')
            return

        if TARGET not in self.df_training:
            logging.debug(' : '+__file__+' : : TARGET column not in df_training...')
            return

        if new_descriptions is None or len(new_descriptions) == 0:
            logging.debug(' : '+__file__+' : : new_descriptions is None or empty')
            return

        logging.info(' : '+__file__+' : : __computeJaros__')

        training_descs = self.df_training[ORIGINAL_DESCRIPTION].values
        training_classes = self.df_training[TARGET].values
        self.predictions = []
        self.probabilities = []

        for i, new_description in enumerate(new_descriptions):
            charge_classes = []
            distances=[]

            for j, training_sent in enumerate(training_descs):
                charge_class = training_classes[j]
                distance = JaroWinklerDistance(new_description, training_sent)

                charge_classes.append(charge_class)
                distances.append(distance)

            index_max = np.argmax(distances)
            self.predictions.append(charge_classes[index_max])
            self.probabilities.append(distances[index_max])
            logging.debug(' : '+__file__+' : : ' + str(i) + ' => ' + new_description + ' = ' + charge_classes[index_max])

    def predict(self, new_descriptions = None):

        """ Returns charge class of closest match.
        Parameters
        ----------
        new_descriptions : array-like of shape = [n_samples]
            The input samples.
        Returns
        -------
        charge_classes : array of strings of shape = [n_samples]
            The label for each new_description is the label of the closest match
            seen during fit.
        probabilities : array of floats of shape = [n_samples]
            The Jaro distance from closest match is returned as probability.
        """
        logging.info(' : '+__file__+' : : predict JaroWrinklerClassifier')
        self.__computeJaros__(new_descriptions)
        return np.array(self.predictions), np.array(self.probabilities)


if __name__ == '__main__':
    print 'Check Jaro:'
    print JaroWinklerDistance('dear', 'bear')