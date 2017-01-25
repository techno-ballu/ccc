import sys
import os
import unittest
from unittest import TextTestRunner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.tests import learning_tests
from app.tests import learning_checks
import runTrainPipe
import time
import logging
from app.conf import appconfig

logging.basicConfig(level=appconfig.LOG_LEVEL ,filename=appconfig.LEARNING_LOG_FILENAME ,filemode='w' ,format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


# This is the main python file that run checks and then
# triggers the python components that initiate the learning
# pipeline.

def runLearningPipeline():
    orig_stdout=sys.stdout
    stdflie=open('../../console_test_learn.txt','w')
    sys.stdout=stdflie
    # resultFile = open('../../learning_log.txt', 'a')
    try:
        logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline run started...\n')
        print 'Starting learning pipeline...'
        runTrainPipe.runTrainingPipeline()
        sys.stdout=orig_stdout
        print 'Learning done successfully :)'
        logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline ran successfully!\n')
    except Exception as e:
        sys.stdout=orig_stdout
        print 'Learning pipeline run was not successful...\n'
        logging.info(' : '+__file__+' : '+' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline run was not successful...\n')
        logging.error(e)
    finally:
        # resultFile.close()
        stdflie.close()
        #TODO uncomment below part - renamed it to console_test_learn
        # os.remove('../../console_test_learn.txt')

noOfTestFailures=0
# Run integration tests here to check if all is well.
check=learning_checks.LearnPipelineChecks()
# Check #1 - check db connection
if not check.isConnectDB():
    print '-'*60
    print 'Connection to database was not successful, Please use valid database name.'
    print 'Or you might not have required permissions to access the database.'
    print 'Enter username and password in DBusername.txt and DBpassword.txt respectively in config folder'
    logging.info(' : '+__file__+' : '+'Connection to database was not successful, Please use valid database name.\nOr you might not have required permissions to access the database.')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60
# Check #2 - training data does not exist
if not check.istrainingDataExists():
    print '-'*60
    print 'There is not enough data in trainig data table Or there is not enough no. of distinct charge classes. Can not proceed with execution of learning pipeline.'
    print 'Please populate data first in training data table.'
    logging.info(' : '+__file__+' : '+'There is not enough data in trainig data table Or there is not enough no. of distinct charge classes. Can not proceed with execution of learning pipeline.Please populate data first in training data table.')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60
# Check #3 - no unlearning/reviewed data
if not check.isnewLearningExists():
    print '-'*60
    print 'No need to run learning pipeline, can proceed with execution of classification pipeline.'
    print 'There is no new unlearning request or reviewed data. Nothing to learn.'
    logging.info(' : '+__file__+' : '+'No need to run learning pipeline, can proceed with execution of classification pipeline.There is no new unlearning request or reviewed data. Nothing to learn.')
    logging.info(' : '+__file__+' : '+time.strftime("%H:%M:%S")+' :: Learning pipeline run was not successful...\n')
    noOfTestFailures+=1
    print '-'*60

# test_suite = unittest.TestLoader().loadTestsFromTestCase(learning_tests.LearningPipelineTests)
# test_result = TextTestRunner().run(test_suite)
# noOfTestFailures = len(test_result.failures)

if noOfTestFailures > 0:
    # print noOfTestFailures, 'learning pipeline tests failed.'
    print 'Exiting now...'
    check.closeConnection()
    sys.exit()

print 'Learning pipeline tests passed!'
runLearningPipeline()

