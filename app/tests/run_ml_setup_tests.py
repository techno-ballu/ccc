import sys
import os
import unittest
from unittest import TextTestRunner

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.tests import ml_setup_tests

def generateMLSetupResultFile(message = 'Machine Learning setup failed.'):
    resultFile = open('../../machine_learning_setup/ML_Checkup_Result.txt', 'w')
    resultFile.write(message)
    resultFile.close()

# Run setup tests here to check if all is well.

# Check #1 - check db connection
# Check #2 - training data does not exist
# Check #3 - no unlearning/reviewed data

test_suite = unittest.TestLoader().loadTestsFromTestCase(ml_setup_tests.MLSetupTests)
test_result = TextTestRunner().run(test_suite)
noOfTestFailures = len(test_result.failures)
generateMLSetupResultFile('Machine Learning setup is successful !!!')
if noOfTestFailures > 0:
    # print noOfTestFailures, 'ML setup tests failed.'
    generateMLSetupResultFile('Machine Learning setup failed.')

