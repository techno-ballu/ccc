import unittest
import pypyodbc
import sys
import os
import shutil
from datetime import datetime
import operator
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.conf import dbconfig
from app.conf import appconfig

class ClassificationPipelineChecks(object):
    def __init__(self):
        self.servername = dbconfig.SERVER_NAME
        self.databasename = dbconfig.DATABASENAME

    def dbConnected(self):
        # check if cursor exists
        try:
            self.cursor
        except AttributeError:
            return False

        return True

    def isConnectDB(self):
        result = False
        try:
            conn_string="Driver={SQL Server};Server="+self.servername+";Database="+self.databasename+";Trusted_Connection=True;"
            self.connection = pypyodbc.connect(conn_string) # , autocommit=True
            self.cursor = self.connection.cursor()
            # print self.connection.getinfo(pypyodbc.SQL_SERVER_NAME), 'connected successfully!'
            result = True
        except:
            # print 'Connection failed...'
            result = False

        return result

    def isModelBinaries(self):
        data_models_learning = appconfig.DATA_MODELS_LEARNING
        models_folder_learning = appconfig.MODELS_FOLDER_LEARNING
        models_folder_classification = appconfig.MODELS_FOLDER_CLASSIFICATION
        number_of_models = appconfig.NUMBER_OF_MODELS
        archived_folder = appconfig.MODELS_FOLDER_ARCHIVED
        datetime_format = appconfig.DATETIME_FORMAT

        if not os.path.exists('../' + data_models_learning):
            return False

        learning_model_list = [s for s in os.listdir('../' + models_folder_learning) if s.endswith('.pickle')]

        if len(learning_model_list) == number_of_models:
            return True
        else:
            # we need to create new datetime archive
            if not os.path.exists('../' + archived_folder):
                return False

            dir_list = os.listdir('../'+archived_folder+'/')
            if len(dir_list) == 0:
                return False

            dates_list = [datetime.strptime(date, datetime_format) for date in dir_list]
            index, value = max(enumerate(dates_list), key=operator.itemgetter(1))
            latest_archive = datetime.strftime(value,datetime_format)
            archived_model_list = [s for s in os.listdir('../'+archived_folder+'/'+latest_archive) if s.endswith('.pickle')]

            if len(archived_model_list) == number_of_models:
                return True
            return False

    def isUnprocessedDecriptions(self):
        if self.dbConnected():
            sql1 = ("select count(*) from [" + self.databasename + "].[dbo].[" + dbconfig.CLAIM_LN_HIST_TAB + "] where PROCESSED_FLAG='Unprocessed' " )
            sql2 = ("select count(*) from [" + self.databasename + "].[dbo].[" + dbconfig.CLSF_OUTPUT_TAB + "] where REV_STAT='Pending' " )
            self.cursor.execute(sql1)
            rows1 = int(self.cursor.fetchone()[0])
            self.cursor.execute(sql2)
            rows2 = int(self.cursor.fetchone()[0])
            row_count = rows1 + rows2
            if row_count > 1:
                return True

        return False

    def closeConnection(self):
        if self.dbConnected():
            self.cursor.close()
            self.connection.close()
            # print 'closed cursor & connection...'

# if __name__ == '__main__':
#     t = unittest.main()
#     print t.result