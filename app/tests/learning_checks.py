import unittest
import pypyodbc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.conf import dbconfig
from app.conf import appconfig

class LearnPipelineChecks(object):
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
            # print self.connection.getinfo(pypyodbc.SQL_DATABASE_NAME)
            # print self.connection.getinfo(pypyodbc.SQL_PROCEDURES)
            # print self.connection.getinfo(pypyodbc.SQL_MAX_DRIVER_CONNECTIONS)
            result = True
        except:
            print 'Connection failed.'
            result = False

        return result

    def istrainingDataExists(self):
        if self.dbConnected():
            # check count(*) instead of fetchAll #done
            sql = ("select count(*) from [" + self.databasename + "].[dbo].[" + dbconfig.TRAINING_TABLE + "]" )
            self.cursor.execute(sql)
            # rows = self.cursor.fetchall()
            # row_count = len(rows)
            row_count = int(self.cursor.fetchone()[0])

            # move 999 to config
            if row_count > appconfig.MIN_TRAINING_SIZE:
                # check # of charge classes also
                sql = ("select distinct CHRG_CLS from ["+self.databasename+"].[dbo].["+dbconfig.TRAINING_TABLE+"]")
                self.cursor.execute(sql)
                rows = self.cursor.fetchall()
                row_count = len(rows)
                if row_count > 1:
                    return True

        return False

    def isnewLearningExists(self):

        # check for model binaries
        if self.isLearningModelsPresent() == False:
            return True

        if self.dbConnected():
            sql_mr = ("select count(*) from [" + self.databasename + "].[dbo].[" + dbconfig.MR_DETAIL_TAB + "] where REV_STAT='Approved' " )
            self.cursor.execute(sql_mr)
            count_mr = int(self.cursor.fetchone()[0])

            sql_unlearn = ("select count(*)  from ["+self.databasename+"].[dbo].["+dbconfig.UNLEARN_MD_TAB+"] where PROCESSED_FLAG='Unprocessed'")
            self.cursor.execute(sql_unlearn)
            count_unlearn = int(self.cursor.fetchone()[0])

            if (count_mr+count_unlearn) > 0:
                return True

        return False

    def isLearningModelsPresent(self):
        models_folder_learning = appconfig.MODELS_FOLDER_LEARNING
        number_of_models = appconfig.NUMBER_OF_MODELS

        if not os.path.exists('../' + models_folder_learning):
            return False

        learning_model_list = [s for s in os.listdir('../' + models_folder_learning) if s.endswith('.pickle')]
        if len(learning_model_list) == number_of_models:
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