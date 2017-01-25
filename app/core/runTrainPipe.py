import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.core import updateTables
from app.conf import appconfig
from app.core import pipeline_train
from app.core import pipeline_UpdateCorpus
import logging

def runTrainingPipeline():
    startTrainPipe=time.time()
    training_file = appconfig.TRAINING_FILE
    data_folder = appconfig.DATA_MODELS_LEARNING


    pipeline_train.manage_folders()
    logging.info(' : '+__file__+' : '+'manage_folders method executed.')
    dbConnection=updateTables.DatabaseConnections()
    pipeline_UpdateCorpus.updateCorpus()
    logging.info(' : '+__file__+' : '+'Abbreviation and spellings corpus updated. ')
    if dbConnection.checkUnlearningRequest()>0:
        dbConnection.doUnlearning()
        dbConnection.changeUnlearningStatus()
        dbConnection.updateUnlearningHistoryTable()
        logging.info(' : '+__file__+' : '+'Unlearning requests processed. ')



    dbConnection.updateTrainingFromManualReview()
    logging.info(' : '+__file__+' : '+'Training data updated from manual review feedback.')
    dbConnection.clearMRTable()


    dbConnection.pullLimitedTrainingData(data_folder,training_file)
    # dbConnection.pullTrainingData(data_folder,training_file)
    logging.info(' : '+__file__+' : '+'Training data fetched from table.')
    metadata=pipeline_train.train_pipeline()
    dbConnection.updateTrainingMD(metadata)
    logging.info(' : '+__file__+' : '+'Training metadata updated.')
    dbConnection.closeConnection()
    print "Total time taken in training Pipeline + database communication:"
    print time.time()-startTrainPipe


if __name__=="__main__":
    runTrainingPipeline()