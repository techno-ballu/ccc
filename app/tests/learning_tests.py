import unittest
import learning_checks

class LearningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tests = learning_checks.LearnPipelineChecks()

    @classmethod
    def tearDownClass(self):
        self.tests.closeConnection()

    def test_db(self):
        self.assertTrue(self.tests.isConnectDB())

    def test_training(self):
        self.assertTrue(self.tests.istrainingDataExists())

    def test_learning(self):
        self.assertTrue(self.tests.isnewLearningExists())

if __name__ == '__main__':
    unittest.main()