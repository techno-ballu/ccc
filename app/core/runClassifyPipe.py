import sys
import os
import shutil
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.core import updateTables
from app.core import pipeline_classify_unlabelled
import time
import subprocess
from app.core import  extractChanges
from app.conf import appconfig
import logging
from datetime import datetime

def runClassificationPipeline():
    normArgument=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../'))).replace('\\','\\\\')+'\\\\'
    startClass=time.time()
    startPull=time.time()
    logging.info(' : '+__file__+' : '+'In runClassificationPipeline() method')
    classification_folder = appconfig.MODELS_FOLDER_CLASSIFICATION
    training_file = appconfig.TRAINING_FILE

    if os.path.exists('../' + classification_folder):
        tmp = tempfile.mktemp(dir=os.path.dirname('../' +classification_folder))
        shutil.move('../' + classification_folder, '../' +tmp)
        shutil.rmtree('../' + tmp)
    os.makedirs('../' + classification_folder)

    # create classification_models folder if not exists
    if not os.path.exists('../' + classification_folder):
        # create the models folder if not exists
        os.makedirs('../' + classification_folder)

    dbConnection=updateTables.DatabaseConnections()
    checkPending=1
    noPending=0
    try:
        noPending=dbConnection.checkPendingClaims()
    except Exception as e:
        checkPending=0
        logging.error(e)

    noRecordsPulled=dbConnection.pullDistinctClaimLines()
    logging.debug(' : '+__file__+' : '+ str(noRecordsPulled) +' Unique descriptios were fetched from clm_ln_hist_tab to classify.')
    abbrConnection=extractChanges.DBconnect()
    abbrConnection.generateAbbrevReverseMapping()
    no_of_chrg_cls=dbConnection.autoClassifySet1()
    lenSet2=dbConnection.pullSet2CompleteToCSV()
    print "Time taken in pulling distinct claim lines and complete set to csv:"
    print time.time()-startPull
    high_conf_covg=1
    if lenSet2>0:
        logging.info(' : '+__file__+' : '+'Normalization of claim lines started')
        startNormal=time.time()
        subprocess.call(['runNormalization.bat',normArgument])
        logging.info(' : '+__file__+' : '+'Normalization done.')
        print "Time taken in Normalization:"
        print time.time()-startNormal
        dbConnection.pullTrainingDataInClassificationPipeline(classification_folder,training_file)
        lenSet2final=dbConnection.autoClassifySet2Similars()
        classification_date = str(datetime.now())
        if lenSet2final>0:
            classification_date,ClassifiedResult=pipeline_classify_unlabelled.classifyUnlabeled()

            dbConnection.uploadClassified(ClassifiedResult)
            no_of_chrg_cls=len(ClassifiedResult['Voted Output'].unique())
            high_conf_covg=len(ClassifiedResult[ClassifiedResult['Confidence Score']>0.9]['Confidence Score'])/float(len(ClassifiedResult['Confidence Score']))
    elif lenSet2==0:
        classification_date = str(datetime.now())





    if checkPending==1:
        dbConnection.deletePendingClaims()
        logging.info(' : '+__file__+' : '+str(noPending)+' already pending unique claim lines were also processed with current batch.')
    dbConnection.mapClaimLineTabToClassification()
    dbConnection.countNoOfRecords()
    dbConnection.changeProcessedStatus()
    dbConnection.putNoRevNecessaryTag()

    total_rec=dbConnection.totalRec
    total_unique=dbConnection.totalUnique
    new_rec=dbConnection.newRec
    unique_new_rec=dbConnection.uniqueNewRec
    metadata=[classification_date,total_rec,no_of_chrg_cls,total_unique,new_rec,unique_new_rec,high_conf_covg]
    dbConnection.updateClassificationMD(metadata)
    dbConnection.closeConnection()
    dbConnection.closeMetaMapServers()
    logging.debug(' : '+__file__+' : '+'Classification metadata: ')
    logging.debug(' : '+__file__+' : '+'Classification date: ' + str(classification_date))
    logging.debug(' : '+__file__+' : '+'total records classified : ' + str(total_rec))
    logging.debug(' : '+__file__+' : '+'No. of charge classed found :: ' + str(no_of_chrg_cls))
    logging.debug(' : '+__file__+' : '+'Total unique classified : ' + str(total_unique))
    logging.debug(' : '+__file__+' : '+'New records (no direct or partial match) : ' + str(new_rec))
    logging.debug(' : '+__file__+' : '+'unique new records : ' + str(unique_new_rec))
    logging.debug(' : '+__file__+' : '+'high confidence (>0.9) : ' + str(high_conf_covg))
    print "Total time taken in classification+ database communications  excluding normalization:"
    print time.time()-startClass

if __name__=="__main__":
    runClassificationPipeline()