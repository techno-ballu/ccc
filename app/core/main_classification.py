import sys
import os
import unittest
from unittest import TextTestRunner
import runClassifyPipe
import time
import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.tests import classification_tests
from app.tests import classification_checks
from app.tests import ml_setup_checks
from app.conf import appconfig
logging.basicConfig(level=appconfig.LOG_LEVEL ,filename=appconfig.CLASSIFICATION_LOG_FILENAME ,filemode='w' ,format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# This is the main python file that run checks and then
# triggers the python components that initiate the classification
# pipeline.

def runClassificationPipeline():
    logging.info(' : '+__file__+' : '+'Classification pipeline started.')
    orig_stdout=sys.stdout
    stdflie=open('../../console_test_classify.txt','w')
    sys.stdout=stdflie
    # resultFile = open('../../classification_log.txt', 'a')
    try:
        print 'Starting classification pipeline...'
        runClassifyPipe.runClassificationPipeline()
        sys.stdout=orig_stdout
        print 'Classification done successfully :)'
        logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline ran successfully!\n')
    except Exception as e:
        logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline run was not successful...\n')
        logging.error(e)
        sys.stdout=orig_stdout
        print 'Classification pipeline run was not successful...\n'
    finally:
        # resultFile.close()
        stdflie.close()
        #TODO uncomment below part - renamed to console_test_classify
        # os.remove('../../console_test_classify.txt')

noOfTestFailures=0
# Run integration tests here to check if all is well.
check=classification_checks.ClassificationPipelineChecks()
mlCheck=ml_setup_checks.MLSetupChecks()
# Check #1 - check db connection
if not check.isConnectDB():
    print '-'*60
    print 'Connection to database was not successful, Please use valid database name.'
    print 'Or you might not have required permissions to access the database.'
    print 'Enter username and password in DBusername.txt and DBpassword.txt respectively in config folder'
    logging.info(' : '+__file__+' : '+'Connection to database was not successful, Please use valid database name.Or you might not have required permissions to access the database.')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60
# Check #2 - check unprocessed data
if not check.isUnprocessedDecriptions():
    print '-'*60
    print 'There is no unprocessed data in source table(CLM_LN_HIST_TAB)'
    print 'No claims to classify.'
    logging.info(' : '+__file__+' : '+'There is no unprocessed data in source table(CLM_LN_HIST_TAB).No claims to classify.')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60

# Check #3 - check if metamap servers are running
if not mlCheck.checkMetaMapServers():
    print '-'*60
    print 'Metamap servers are not running'
    logging.info(' : '+__file__+' : '+'Metamap servers are not running')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60

# Check #4 - check if model binaries are present
if not check.isModelBinaries():
    print '-'*60
    print 'Model binaries are not present neither in classification_models folder nor in archived_models folder'
    print 'Run learning pipeline first, then classification pipeline after its completion .'
    logging.info(' : '+__file__+' : '+'Model binaries are not present neither in classification_models folder nor in archived_models folder.Run learning pipeline first, then classification pipeline after its completion .')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Classification pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60

#
# test_suite = unittest.TestLoader().loadTestsFromTestCase(classification_tests.ClassificationPipelineTests)
# test_result = TextTestRunner().run(test_suite)
#
# noOfTestFailures = len(test_result.failures)

if noOfTestFailures > 0:
    # print noOfTestFailures, 'classification pipeline tests failed.'
    print 'Exiting now...'
    check.closeConnection()
    sys.exit()

print 'Classification pipeline tests passed!'
runClassificationPipeline()

