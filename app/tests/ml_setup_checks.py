import sys
import os
import subprocess
import importlib

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
# from app.conf import appconfig

class MLSetupChecks(object):
    def __init__(self):
        self.imports = {'wikipedia': 'wikipedia'}

    def isPythonImports(self):
        try:
            import wikipedia
            from nltk.corpus import stopwords
            from nltk.stem.porter import PorterStemmer
            from nltk.stem import WordNetLemmatizer
            from nltk.util import ngrams
            from nltk.tokenize import sent_tokenize
            import xgboost
            import gensim
            import sknn
            import pypyodbc
            # import fuzzywuzzy
            return True
        except Exception:
            print 'python imports failed'
            return False
        # for i in self.imports:
        #     try:
        #         importlib.import_module(i)
        #         return True
        #     except Exception:
        #         print 'Python import', i, 'failed...'
        #         return False

    def checkMetaMapServers(self):
        iterator = 1
        while iterator < 5:
            if self.checkIfMetaMapServersAreRunning():
                return True
            else:
                self.startMetaMapServers()

            iterator += 1
        print 'Unable to start MetaMap servers.....!!!!'
        return False


    def checkIfMetaMapServersAreRunning(self):
        # self.mm_path = appconfig.METAMAP_INSTALLATION_FOLDER
        self.mm_path = 'C:/public_mm'
        if not os.path.exists(self.mm_path):
            print 'MetaMap not installed...!!'
            return False
        filepath=self.mm_path + "/bin/testapi.bat"
        p = subprocess.Popen([filepath,'heart attack'], shell=True, stdout = subprocess.PIPE)
        stdout, stderr = p.communicate()
        outLines = stdout.splitlines()
        out = p.returncode  # is 0 if success
        if (out == 0) and (outLines[len(outLines)-1] == '   Negation Status: 0'):
            print 'Servers already running....!!'
            return True
        return False

    def startMetaMapServers(self):
        print 'Restarting servers....'
        current_dir=os.path.dirname( __file__ )
        #os.chdir('../')
        normArgument = self.mm_path.replace('/',"\\")
        current_dir = current_dir.replace('\\',"\\\\")
        #print normArgument
        #print current_dir
        #try:
            #subprocess.call('cd ".\\tests\\"')
            #print os.getcwd()
        #print current_dir
        subprocess.call([current_dir + '\\StartAllservers.bat',normArgument])
        #print 'test end'
        #finally:
        #   os.chdir(current_dir)


    def closeMetaMapServers(self):
        from win32com.client import GetObject
        # mm_path = appconfig.METAMAP_INSTALLATION_FOLDER
        mm_path = 'C:/public_mm'
        mm_path = mm_path.replace('/',"\\")
        servers = ['"'+mm_path+'\\bin\\skrmedpostctl_start.bat"','"'+mm_path+'\\bin\\wsdserverctl_start.bat"','"'+mm_path+'\\bin\\mmserver14.bat"']
        WMI = GetObject('winmgmts:')
        processes = WMI.InstancesOf('Win32_Process')
        for p in WMI.ExecQuery('select * from Win32_Process where Name="cmd.exe"'):
            command = p.Properties_('CommandLine').Value
            if command is None:
                continue
            cmd_tokens = command.split()

            if cmd_tokens[len(cmd_tokens)-1] in servers:
                print cmd_tokens[len(cmd_tokens)-1]
                print "Killing PID:", p.Properties_('CommandLine').Value
                os.system("taskkill /pid "+str(p.Properties_('ProcessId').Value))

# if __name__ == '__main__':
#     t = unittest.main()
#     print t.result
