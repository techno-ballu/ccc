import unittest
from ml_setup_checks import MLSetupChecks

class MLSetupTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tests = MLSetupChecks()

    @classmethod
    def tearDownClass(self):
        self.tests.closeMetaMapServers()

    def test_metamap(self):
        self.assertTrue(self.tests.checkMetaMapServers())

    def test_imports(self):
        self.assertTrue(self.tests.isPythonImports())

if __name__ == '__main__':
    unittest.main()