import unittest
from classification_checks import ClassificationPipelineChecks
from ml_setup_checks import MLSetupChecks

class ClassificationPipelineTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.class_tests = ClassificationPipelineChecks()
        self.setup_tests = MLSetupChecks()

    @classmethod
    def tearDownClass(self):
        self.class_tests.closeConnection()

    def test_metamap(self):
        self.assertTrue(self.setup_tests.checkMetaMapServers())

    def test_db(self):
        self.assertTrue(self.class_tests.isConnectDB())

    def test_binaries(self):
        self.assertTrue(self.class_tests.isModelBinaries())

    def test_unprocessed(self):
        self.assertTrue(self.class_tests.isUnprocessedDecriptions())

if __name__ == '__main__':
    unittest.main()