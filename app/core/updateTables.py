import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pypyodbc
import pandas as pd
import numpy as np
from app.conf import appconfig
from datetime import datetime
from app.conf import dbconfig
from app.core import data_preprocessor
import time
import logging
import re
from sklearn.cross_validation import StratifiedShuffleSplit

def temp_uploadTestData():
    SQL_write="Driver={SQL Server};Server="+dbconfig.SERVER_NAME+";Database=CE_Work;Trusted_Connection=True;"
    write_connection = pypyodbc.connect(SQL_write)
    cursor = write_connection.cursor()
    # data=pd.read_csv("../Test_8batches.csv" ,na_filter=False)
    data = pd.read_excel("../Test_8batches.xlsx")
    # print data.columns # 'ST_DOS','CLM_RCV_DT','EXP_DT'     ST_DOS,CLM_RCV_DT,EXP_DT,
    data=data[['CLM_ID','CLM_LN_ID','ST_DOS','CLM_RCV_DT','EXP_DT','ORIG_DESC','Batch','PROCESSED_FLAG','FACILITY_NM','PATIENT_FIRST_NAME','PATIENT_LAST_NAME']]
    data = data[data['Batch']=='B5']
    print len(data)
    data = data.drop(['Batch'],1)
    print data.columns
    SQL_query=("INSERT INTO CLAIM_LINE_LANDING_TAB_TEMP"
                 "(CLM_ID,CLM_LN_ID,ST_DOS,CLM_RCV_DT,EXP_DT,ORIG_DESC,PROCESSED_FLAG,FACILITY_NM,PATIENT_FIRST_NAME,PATIENT_LAST_NAME)"
                 "VALUES (?,?,?,?, ?,?,?,?, ?,?)")
    dataList=data.values.tolist()
    cursor.executemany(SQL_query, dataList)
    cursor.commit()
    SQL_query=("INSERT INTO CLAIM_LINE_LANDING_TAB "
                 " SELECT * FROM CLAIM_LINE_LANDING_TAB_TEMP")
    cursor.execute(SQL_query)
    cursor.commit()
    cursor.close()
    write_connection.close()

class DatabaseConnections(object):
    def __init__(self):
        self.datafolder=appconfig.DATA_MODELS_LEARNING
        self.word2vecfolder=appconfig.WORD2VEC_FOLDER
        self.classificationfolder = appconfig.MODELS_FOLDER_CLASSIFICATION
        # self.outputfolder=appconfig.OUTPUT_FOLDER
        self.predictionfile='AllTrainingFinal.csv'
        self.unlabeledfile=appconfig.UNLABELED_FILE
        self.important_words_file=appconfig.IMPORTANT_WORDS_FILE
        self.text_feature=appconfig.TEXT_FEATURE
        self.onedaydatafile=appconfig.ONE_DAY_DATA
        self.character_encoding=appconfig.ENCODING_TO_USE
        self.columnsTrain=appconfig.TRAINING_COLUMNS_TO_UPLOAD
        self.columnTrain_fetch=appconfig.TRAINING_COLUMNS_TO_FETCH
        self.servername=dbconfig.SERVER_NAME
        self.databasename=dbconfig.DATABASENAME
        self.trainingTable=dbconfig.TRAINING_TABLE
        self.trainingMDTable=dbconfig.TRAINING_MD
        self.claimLineHistoryTable=dbconfig.CLAIM_LN_HIST_TAB
        self.origDescTable=dbconfig.DISTINCT_ORIG_DESC_TAB
        self.clsfTempTable=dbconfig.CLASSIFICATION_TEMP_TAB
        self.classfMDTable=dbconfig.CLASSF_MD_TAB
        self.classificationOutputTable=dbconfig.CLSF_OUTPUT_TAB
        self.unlearnDataTable=dbconfig.UNLEARN_DATA_TAB
        self.unlearnMDTable=dbconfig.UNLEARN_MD_TAB
        self.mrDetailTable=dbconfig.MR_DETAIL_TAB
        self.thresholdCLMLineTable=dbconfig.THRESHOLD_CONFIG
        self.unlearnHistoryTable=dbconfig.UNLEARN_HIS_TAB
        self.medicalConceptsTable=dbconfig.MEDICAL_CONCEPTS_TABLE
        self.cc_algorithm_result_table = dbconfig.CC_ALGORITHM_RESULT_TABLE
        self.thresholdClaimLineDescipInDB=dbconfig.THRESHOLD_CLAIM_LINE_DESCRIP_IN_DB
        self.totalRec=None
        self.totalUnique=None
        self.totalSet1=None
        self.uniqueSet1=None
        self.newRec=None
        self.uniqueNewRec=None
        self.set2Final=None
        self.set2Similars=None
        self.SQL_write="Driver={SQL Server};Server="+self.servername+";Database="+self.databasename+";Trusted_Connection=True;"
        self.write_connection = pypyodbc.connect(self.SQL_write)
        self.cursor = self.write_connection.cursor()

    def closeConnection(self):
        self.cursor.close()
        self.write_connection.close()

    def updateTrainData(self):
        logging.info(' : '+__file__+' : '+'In updateTrainData() method.')
        startTrainUpdate=time.time()
        data=pd.read_csv("../"+self.word2vecfolder+"/"+self.predictionfile ,na_filter=False)
        data=data[self.columnsTrain]
        data.insert(loc=4,column='Date',value=datetime.now())
        SQL_query=("INSERT INTO "+self.trainingTable+" "
                     "(ORIG_DESC,"
                        "NORM_DESC, CHRG_CLS, FLAG,"
                        "CREATE_DT, FTR1, FTR2,FTR3,FTR4)"
                     "VALUES (?,?,?,?,convert(date,?),?,?,?,?)")


        test=data.values.tolist()
        self.cursor.executemany(SQL_query, test)
        self.cursor.commit()
        print "Time taken in updating train data:"
        print time.time()-startTrainUpdate

    def upsampleMinorChargeClassesinUpdateTable(self, min_count,data):
        logging.info(' : '+__file__+' : In upsampleMinorChargeClassesinUpdateTable()')
        chrgClass_valueCounts = data[appconfig.TARGET].value_counts()
        logging.debug(' : '+__file__+' : '+' Checking training data for upsampling')
        if min(chrgClass_valueCounts) < min_count:
            vc = chrgClass_valueCounts[chrgClass_valueCounts < min_count]
            for i in range(0,len(vc)):
                chrgClass = vc.index[i]
                count = vc[chrgClass]
                logging.debug(' : '+__file__+' : ' + chrgClass + ' needs to be upsampled.')
                df_chrgClass = data[data[appconfig.TARGET]==chrgClass]
                n = int(min_count/count)
                if n > 1:
                   df_chrgClass = df_chrgClass.append([df_chrgClass]*(n-1),ignore_index=True)
                df_chrgClass = df_chrgClass.ix[np.random.choice(df_chrgClass.index, min_count-count)]
                data = data.append(df_chrgClass,ignore_index=True)
                # data['training'] =  self.df_training
            return data

        return data

    def pullTrainingData(self,data_folder,trainingFile):
        logging.info(' : '+__file__+' : '+'In pullTrainingData() method.')
        startPullTrain=time.time()
        SQL_query=("SELECT * FROM "+self.trainingTable+" ")

        self.cursor.execute(SQL_query)
        trainigData=pd.DataFrame(self.cursor.fetchall(),columns=self.columnTrain_fetch)
        print 'Length of Complete training data: '
        lenTrain=len(trainigData)
        logging.debug(' : '+__file__+' : '+'Length of Complete training data: '+str(lenTrain))
        logging.debug(' : '+__file__+' : '+'Number of training lines to sample: '+str(appconfig.NO_TRAINING_LINES))
        buffer=appconfig.CHARGE_CLASS_BUFFER
        if lenTrain > (int(appconfig.NO_TRAINING_LINES) + buffer):
            logging.debug(' : '+__file__+' : '+' Training data length exceeds limit!')

            trainigData=self.upsampleMinorChargeClassesinUpdateTable(appconfig.MIN_SAMPLE_SIZE,trainigData)
            y = trainigData[appconfig.TARGET].values
            logging.info(' : '+__file__+' : '+'Length after upsampling: '+ str(len(trainigData)))
            # spilt into stratified training & test sets
            logging.debug(' : '+__file__+' : Stratified splitting to get sampled training data of lines = ' + str(appconfig.NO_TRAINING_LINES))
            sss = StratifiedShuffleSplit(y, 1, train_size=int(appconfig.NO_TRAINING_LINES), random_state=786)
            train_index, test_index = list(sss)[0]
            trainigData = trainigData.iloc[train_index]
            logging.debug(' : '+__file__+' : '+'Length of training data after sampling: '+str(len(trainigData)))

        self.cursor.commit()
        trainigData.to_csv('../'+data_folder+'/'+trainingFile,index=False,encoding=self.character_encoding)
        print "time taken in pulling training data"
        print time.time()-startPullTrain

    def pullLimitedTrainingData(self,data_folder,trainingFile):
        logging.info(' : '+__file__+' : '+'In pullLimitedTrainingData() method.')
        startPullTrain=time.time()

        num_of_training_lines = str(appconfig.NO_TRAINING_LINES)
        SQL_query = ("SELECT top " + num_of_training_lines+" * FROM ["+self.databasename+"].[dbo].["+self.trainingTable+"] order by create_dt desc")
        # SQL_query=("SELECT * FROM "+self.trainingTable+" ")
        logging.debug(' : '+__file__+' : '+SQL_query)
        self.cursor.execute(SQL_query)
        trainingData = pd.DataFrame(self.cursor.fetchall(),columns=self.columnTrain_fetch)
        print 'Length of Complete training data: '
        lenTrain=len(trainingData)
        logging.debug(' : '+__file__+' : '+'Length of training data: '+str(lenTrain))
        # logging.debug(' : '+__file__+' : '+'Number of training lines to sample: '+str(appconfig.NO_TRAINING_LINES))
        # buffer=appconfig.CHARGE_CLASS_BUFFER
        # if lenTrain > (int(appconfig.NO_TRAINING_LINES) + buffer):
        #     logging.debug(' : '+__file__+' : '+' Training data length exceeds limit!')
        #
        #     trainigData=self.upsampleMinorChargeClassesinUpdateTable(appconfig.MIN_SAMPLE_SIZE,trainigData)
        #     y = trainigData[appconfig.TARGET].values
        #     logging.info(' : '+__file__+' : '+'Length after upsampling: '+ str(len(trainigData)))
        #     # spilt into stratified training & test sets
        #     logging.debug(' : '+__file__+' : Stratified splitting to get sampled training data of lines = ' + str(appconfig.NO_TRAINING_LINES))
        #     sss = StratifiedShuffleSplit(y, 1, train_size=int(appconfig.NO_TRAINING_LINES), random_state=786)
        #     train_index, test_index = list(sss)[0]
        #     trainigData = trainigData.iloc[train_index]
        #     logging.debug(' : '+__file__+' : '+'Length of training data after sampling: '+str(len(trainigData)))

        trainingData.to_csv('../'+data_folder+'/'+trainingFile,index=False,encoding=self.character_encoding)
        print "time taken in pulling training data"
        print time.time()-startPullTrain

    def pullTrainingDataInClassificationPipeline(self,data_folder,trainingFile):
        logging.info(' : '+__file__+' : '+'In pullTrainingDataInClassificationPipeline() method.')
        startPullTrain=time.time()
        SQL_query=("SELECT * FROM "+self.trainingTable+" ")

        self.cursor.execute(SQL_query)
        trainigData=pd.DataFrame(self.cursor.fetchall(),columns=self.columnTrain_fetch)
        print 'Length of Complete training data: '
        lenTrain=len(trainigData)
        print lenTrain
        logging.debug(' : '+__file__+' : '+'Length of Complete training data: '+str(lenTrain))
        self.cursor.commit()
        trainigData.to_csv('../'+data_folder+'/'+trainingFile,index=False,encoding=self.character_encoding)
        print "time taken in pulling training data"
        print time.time()-startPullTrain

    def pullMedicalConcepts(self,data_folder,meadicalconceptsfile):
        logging.info(' : '+__file__+' : '+'In pullMedicalConcepts() method.')
        SQL_query=("SELECT * FROM "+self.medicalConceptsTable+" ")

        self.cursor.execute(SQL_query)
        medicalconcepts=pd.DataFrame(self.cursor.fetchall(),columns=['Concept','word2vec key words'])
        medicalconcepts.to_csv('../'+data_folder+'/'+meadicalconceptsfile,index=False,encoding=self.character_encoding)


    def pullDistinctClaimLines(self):
        logging.info(' : '+__file__+' : '+'In pullDistinctClaimLines() method.')
        startpullDistinct=time.time()
        SQL_query1=("update ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"]"
                        " set ORIG_DESC=rtrim(ltrim(replace(replace(replace(replace(orig_desc,char(34),' '),',',' '),char(13),' '),char(10),' ')))")
        SQL_query2=("TRUNCATE TABLE ["+self.databasename+"].[dbo].["+self.origDescTable+"]"
        " INSERT INTO ["+self.databasename+"].[dbo].["+self.origDescTable+"]"
            " (ORIG_DESC)"
            " SELECT  distinct ORIG_DESC FROM   ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] where PROCESSED_FLAG='Unprocessed' ")
        SQL_queryRecordsPulled=("select count (*) from ["+self.databasename+"].[dbo].["+self.origDescTable+"]")

        self.cursor.execute(SQL_query1)
        self.cursor.commit()
        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        self.cursor.execute(SQL_queryRecordsPulled)
        noRecordsPulled=int(self.cursor.fetchone()[0])
        self.cursor.commit()
        print "time taken in pulling distinct claim lines:"
        print time.time()-startpullDistinct
        return noRecordsPulled

    def countNoOfRecords(self):
        logging.info(' : '+__file__+' : '+'In countNoOfRecords() method.')
        startCount=time.time()
        # self.SQL_write="Driver={SQL Server};Server="+self.servername+";Database="+self.databasename+";Trusted_Connection=True;"
        # self.write_connection = pypyodbc.connect(self.SQL_write)
        # cursor = self.write_connection.cursor()
        # SQL_query=("TRUNCATE TABLE ["+self.databasename+"].[dbo].[CLASSIFICATION_TAB] INSERT INTO ["+self.databasename+"].[dbo].[CLASSIFICATION_TAB] (CLASSF_DT,ORIG_DESC, NORM_DESC, CHRG_CLS, CONF_SCORE, FTR1, FTR2, FTR3, FTR4) SELECT  getdate(),ORIG_DESC, NORM_DESC, CHRG_CLS, CONF_SCORE, FTR1, FTR2, FTR3, FTR4 FROM    (SELECT ORIG_DESC, NORM_DESC, CHRG_CLS, CONF_SCORE, FTR1, FTR2, FTR3, FTR4, ROW_NUMBER() OVER (PARTITION BY ORIG_DESC order by CONF_SCORE) AS RowNumber FROM   ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"_COPY] ) AS a WHERE   a.RowNumber = 1")
        SQL_query1=("select count(*) from ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] where PROCESSED_FLAG='Unprocessed'")
        SQL_query2=("select count(*) from ["+self.databasename+"].[dbo].["+self.origDescTable+"] ")
        SQL_query3=("select count(*) "
                    " from ["+self.databasename+"].[dbo].["+self.origDescTable+"] b"
                    " LEFT JOIN ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train ON train.ORIG_DESC=b.ORIG_DESC "
                    " WHERE train.[ORIG_DESC] IS NULL")
        SQL_query4=("select count(*) "
                    " from ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] b"
                    " LEFT JOIN ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train ON train.ORIG_DESC=b.ORIG_DESC "
                    " WHERE train.[ORIG_DESC] IS NULL and b.[PROCESSED_FLAG]='Unprocessed'")
        # SQL_query5=("select count(*) "
        #             " from ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] b"
        #             " LEFT JOIN ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train ON train.ORIG_DESC=b.ORIG_DESC "
        #             " WHERE train.[ORIG_DESC] IS NULL")
        # SQL_query6=("select count(*) "
        #             " from ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] b"
        #             " LEFT JOIN ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train ON train.ORIG_DESC=b.ORIG_DESC "
        #             " WHERE train.[ORIG_DESC] IS NOT NULL")
        self.cursor.execute(SQL_query1)
        self.totalRec=int(self.cursor.fetchone()[0])
        self.cursor.execute(SQL_query2)
        self.totalUnique=int(self.cursor.fetchone()[0])
        self.cursor.execute(SQL_query3)
        self.uniqueNewRec=int(self.cursor.fetchone()[0])
        self.cursor.execute(SQL_query4)
        self.newRec=int(self.cursor.fetchone()[0])
        self.uniqueSet1=self.totalUnique-self.uniqueNewRec
        self.totalSet1=self.totalRec-self.newRec
        print "Time taken in counting:"
        print time.time()-startCount

        # self.cursor.close()
        # self.write_connection.close()


    def uploadClassified(self,algoResult):
        logging.info(' : '+__file__+' : '+'In uploadClassified() method.')
        startUpload=time.time()

        result=algoResult[appconfig.CLASSIFICATION_COLUMN_TO_UPLOAD]
        algoResult = algoResult[['Original Description','Normalized Description','Voted Output','Confidence Score','Predicted_Class_Jaro','Prob_Jaro','Predicted_Class_DLevenshtein','Prob_DLevenshtein','Predicted_Class_Logistic_OvR','Prob_Logistic_OvR','Predicted_Class_DeepNN','Prob_DeepNN']]
        SQL_query=("INSERT INTO ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]"
        " ([CLASSF_DT],[ORIG_DESC],[NORM_DESC],[CHRG_CLS],[CONF_SCORE],[FTR1],[FTR2],[FTR3],[FTR4],[CHNGS]) "
                     " VALUES (?,?,?,?,?,?,?,?,?,?)")
        SQL_query2=("update ["+self.databasename+"].[dbo].["+self.clsfTempTable+"] set orig_desc=rtrim(ltrim(orig_desc))")
        SQL_query3 =("INSERT INTO ["+self.databasename+"].[dbo].["+self.cc_algorithm_result_table+"]"
        "  ([TIME_STAMP],[ORIG_DESC],[NORM_DESC],[OUTPUT_CHARGE_CLASS],[OUTPUT_CONFIDENCE_SCORE],[JAROWINKLER_CHARGE_CLASS],[JAROWINKLER_CONFIDENCE_SCORE],"
        " [DAMERAU_LEVENSHTEIN_CHARGE_CLASS],[DAMERAU_LEVENSHTEIN_CONFIDENCE_SCORE],[LOGISTIC_REGRESSION_CHARGE_CLASS]"
        " ,[LOGISTIC_REGRESSION_CONFIDENCE_SCORE],[NEURAL_NET_CHARGE_CLASS],[NEURAL_NET_CONFIDENCE_SCORE]) "
                     " VALUES (getdate(),?,?,?,?,?,?,?,?,?,?,?,?)")
        logging.info("before replace")
        result=result.replace(np.nan,'',regex=True)
        algoResult=algoResult.replace(np.nan,'',regex=True)
        logging.info("after replace NaN")
        result=result.values.tolist()
        algoResult=algoResult.values.tolist()
        self.cursor.executemany(SQL_query, result)
        self.cursor.commit()
        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        self.cursor.executemany(SQL_query3, algoResult)
        self.cursor.commit()
        logging.info('after executing query')
        print "Time taken in uploading Classification results:"
        print time.time()-startUpload

    def updateTrainingMD(self,metadata):
        logging.info(' : '+__file__+' : '+'In updateTrainingMD() method.')
        startUpdateTrainingMD=time.time()
        SQL_query=("INSERT INTO "+self.trainingMDTable+" "
                     "(TRNG_DT,"
                        "NO_OF_TRAIN_REC, NO_OF_TEST_REC,NO_OF_CHRG_CLS, ONL_RETRO_RATIO,"
                        "NO_OF_BOW_TOTAL,NO_OF_W2V_TOTAL,NO_OF_SEL_FEAT,"
                        "CLASSIFIERS,STD_CUTOFF,CORRELATION_CUTOFF)"
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?)")

        self.cursor.execute(SQL_query, metadata)
        self.cursor.commit()
        print "Time taken in uploading training MD"
        print time.time()-startUpdateTrainingMD


    def updateClassificationMD(self,metadata):
        logging.info(' : '+__file__+' : '+'In updateClassificationMD() method.')
        SQL_query=("INSERT INTO "+self.classfMDTable+" "
                     " (clsf_dt,"
                        " no_of_rec,no_of_chrg_cls,unique_rec,new_rec,unique_new_rec,HIGH_CONF_COVG)"
                     " VALUES (?,?,?,?,?,?,?)")


        self.cursor.execute(SQL_query, metadata)
        self.cursor.commit()


    def autoClassifySet1(self):
        logging.info(' : '+__file__+' : '+'In autoClassifySet1() method.')
        startSet1=time.time()
        SQL_queryFetchOrigDesc=("TRUNCATE TABLE ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]"
                                " truncate table ["+self.databasename+"].[dbo].["+self.cc_algorithm_result_table+"] select * from "
                                " ["+self.databasename+"].[dbo].["+self.origDescTable+"]")

        self.cursor.execute(SQL_queryFetchOrigDesc)
        allOrigDesc=pd.DataFrame(self.cursor.fetchall(),columns=[appconfig.ORIGINAL_DESCRIPTION])
        self.cursor.commit()
        allOrigDesc['Orig_desc_copy'] = allOrigDesc[appconfig.ORIGINAL_DESCRIPTION].apply(lambda x: re.sub('[^A-Za-z0-9+%.>/&]',' ',x))
        allOrigDesc['Orig_desc_copy'] = allOrigDesc['Orig_desc_copy'].apply(lambda x: re.sub('[ ]{2,}',' ',x))
        SQL_queryUploadInTemp=(" create table #temp (orig_desc varchar(600) not null,orig_desc_copy varchar(600) not null) INSERT INTO #temp"
        " ([ORIG_DESC],orig_desc_copy)"
                     "VALUES (?,?)"
            " INSERT INTO ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]"
            " ([CLASSF_DT],[ORIG_DESC],[NORM_DESC],[CHRG_CLS],[CONF_SCORE],[FTR1],[FTR2],[FTR3],[FTR4])"
            " select CLASSF_DT=getdate(),b.ORIG_DESC, train.NORM_DESC, train.CHRG_CLS, CONF_SCORE=1, train.FTR1, train.FTR2, train.FTR3, train.FTR4 "
            " from #temp b"
            " LEFT JOIN ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train ON train.ORIG_DESC=b.orig_desc_copy "
            " WHERE train.[ORIG_DESC] IS NOT NULL"
            " drop table #temp")

        SQL_queryAlgoresult = (" INSERT INTO ["+self.databasename+"].[dbo].["+self.cc_algorithm_result_table+"]"
            " ([TIME_STAMP],[ORIG_DESC],[NORM_DESC],[OUTPUT_CHARGE_CLASS],[OUTPUT_CONFIDENCE_SCORE])"
            " select classf_dt,b.ORIG_DESC, b.NORM_DESC, b.CHRG_CLS, b.CONF_SCORE "
            " from ["+self.databasename+"].[dbo].["+self.clsfTempTable+"] b")

        SQL_queryUniqueClass=("select count(distinct chrg_cls)from ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]")

        SQL_set1=("select count(*)from ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]")

        allOrigDesc=allOrigDesc.values.tolist()
        self.cursor.executemany(SQL_queryUploadInTemp, allOrigDesc)
        self.cursor.commit()
        self.cursor.execute(SQL_queryUniqueClass)
        no_of_chrg_cls=float(self.cursor.fetchone()[0])
        self.cursor.commit()
        self.cursor.execute(SQL_queryAlgoresult)
        self.cursor.commit()
        self.cursor.execute(SQL_set1)
        lenSet1=int(self.cursor.fetchone()[0])
        logging.debug(' : '+__file__+' : '+'Length of set1 (direct matches): '+str(lenSet1))
        self.cursor.commit()
        print "Time taken in auto classifying set1:"
        print time.time()-startSet1
        return no_of_chrg_cls


    def pullSet2CompleteToCSV(self):
        logging.info(' : '+__file__+' : '+'In pullSet2CompleteToCSV() method.')
        startset2CSV=time.time()
        SQL_query=(" select b.ORIG_DESC "
        " from ["+self.databasename+"].[dbo].["+self.origDescTable+"] b"
        " LEFT JOIN ["+self.databasename+"].[dbo].["+self.clsfTempTable+"] as train ON train.ORIG_DESC=b.ORIG_DESC "
        " WHERE train.[ORIG_DESC] IS NULL")

        self.cursor.execute(SQL_query)
        set2=pd.DataFrame(self.cursor.fetchall(),columns=['Original Description'])
        self.cursor.commit()
        print 'length of set 2 '
        print len(set2)
        print "Time taken in pulling and writing complete set2 to csv:"
        logging.debug(' : '+__file__+' : '+'Length of set2 complete : '+str(len(set2)))
        set2.to_csv('../'+self.classificationfolder+'/'+self.onedaydatafile,index=False,encoding=self.character_encoding)
        print time.time()-startset2CSV
        return len(set2)
        
    def autoClassifySet2Similars(self):
        logging.info(' : '+__file__+' : '+'In autoClassifySet2Similars() method.')
        self.set2Final=pd.read_csv('../'+self.classificationfolder+'/'+self.unlabeledfile)
        trainingData=pd.read_csv('../' + self.classificationfolder +'/' + appconfig.TRAINING_FILE)
        preprocessor=data_preprocessor.DataPreprocessor(None,None,None,None,None)
        startSimilar=time.time()
        self.set2Final,self.set2Similars= preprocessor.setSimilarToTrainingNew(self.set2Final,trainingData)
        logging.debug(' : '+__file__+' : '+'Length of partial matches : '+str(len(self.set2Similars)))
        logging.debug(' : '+__file__+' : '+'Length of set2 complete : '+str(len(self.set2Final)))
        print "Time taken in identifying set2.1:"
        print time.time()-startSimilar
        startFeatureExtraction=time.time()
        print "time taken in feature extraction for set2.1 and uploading to output table:"
        # print self.set2Similars
        if len(self.set2Similars)>0:
            self.set2Final.to_csv('../'+self.classificationfolder+'/'+self.unlabeledfile,index=False)
            from app.core import learner
            from app.core import classifier
            classification_engine = classifier.ClassificationEngine(None)
            imp_words = classification_engine.loadImportantWords(self.important_words_file)
            feature_impo=learner.FeatureImportance(imp_words)
            norm_desc=self.set2Similars[[self.text_feature]]
            self.set2Similars=feature_impo.extractFeatures(self.set2Similars,self.text_feature)
            self.set2Similars=self.set2Similars.replace(np.nan,'',regex=True)
            self.set2Similars[[self.text_feature]]=norm_desc
            self.set2Similars=self.set2Similars[['Original Description','Normalized Description','Charge Class','Confidence Score','Feature 1', 'Feature 2', 'Feature 3','Feature 4','Changes']]
            SQL_query=("INSERT INTO ["+self.databasename+"].[dbo].["+self.clsfTempTable+"]"
            " ([CLASSF_DT],[ORIG_DESC],[NORM_DESC],[CHRG_CLS],[CONF_SCORE],[FTR1],[FTR2],[FTR3],[FTR4],[CHNGS])"
                         "VALUES (getdate(),?,?,?,?,?,?,?,?,?)")
            SQL_query2=("INSERT INTO ["+self.databasename+"].[dbo].["+self.cc_algorithm_result_table+"]"
            " ([TIME_STAMP],[ORIG_DESC],[NORM_DESC],[OUTPUT_CHARGE_CLASS],[OUTPUT_CONFIDENCE_SCORE],[JAROWINKLER_CHARGE_CLASS],[JAROWINKLER_CONFIDENCE_SCORE])"
                         "VALUES (getdate(),?,?,?,?,?,?)")

            print "Classifying set3"
            algo_result = self.set2Similars[['Original Description','Normalized Description','Charge Class','Confidence Score']]
            algo_result['Jaro Prediction'] = algo_result['Charge Class']
            algo_result['Jaro Score'] = algo_result['Confidence Score']
            data2 = algo_result.values.tolist()
            data=self.set2Similars.values.tolist()
            # print data
            self.cursor.executemany(SQL_query,data)
            self.cursor.commit()
            self.cursor.executemany(SQL_query2,data2)
            self.cursor.commit()
        print time.time()-startFeatureExtraction
        return len(self.set2Final)


    def mapClaimLineTabToClassification(self):
        logging.info(' : '+__file__+' : '+'In mapClaimLineTabToClassification() method.')
        startJoin=time.time()
        SQL_query2=(" Insert into ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"]"
                " ([CLM_ID],[CLM_LN_ID],[ORIG_DESC],[NORM_DESC],[CHRG_CLS],[PATIENT_NM],[CLIENT_NM],[PROV_NM],[ST_DOS]"
                " ,[CLM_RCV_DT],[EXP_DT],[CLM_AMT],[FTR1],[FTR2],[FTR3],[FTR4],[CONF_SCORE],[REV_STAT],[CHNGS],[batch_date],[CreatedBy],[CreatedDate],[NORM_DESC_ORIG],[ReceivedDate])"
                " select a.[CLM_ID],a.[CLM_LN_ID],a.[ORIG_DESC],b.[NORM_DESC],b.[CHRG_CLS],concat(a.patient_last_name,', ',a.patient_first_name),a.[CLIENT_NM],a.[PROV_NM],a.[ST_DOS]"
                " ,a.[CLM_RCV_DT],[EXP_DT]=DATEADD(day,3,a.[CLM_RCV_DT]),a.[CLM_AMT],b.[FTR1],b.[FTR2],b.[FTR3],b.[FTR4],"
                " b.[CONF_SCORE],[REV_STAT]='Pending',b.[CHNGS],a.[batch_date],current_user,getdate(),b.[NORM_DESC],a.[ReceivedDate]"
                " from ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] a"
                " LEFT JOIN ["+self.databasename+"].[dbo].["+self.clsfTempTable+"] as b ON b.ORIG_DESC=a.ORIG_DESC"
                "  where a.[PROCESSED_FLAG]='Unprocessed'")


        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        print "Time taken in joining output table to input:"
        print time.time()-startJoin


    def changeProcessedStatus(self):
        logging.info(' : '+__file__+' : '+'In changeProcessedStatus() method.')
        startProcessed=time.time()
        SQL_query=(" update  ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"]"
                " set [PROCESSED_FLAG]='Processed'"
                " from ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"] a, ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] as b"

                "  where b.[PROCESSED_FLAG]='Unprocessed' and a.orig_desc=b.orig_desc and a.clm_id=b.clm_id and a.clm_ln_id=b.clm_ln_id and a.ReceivedDate=b.ReceivedDate")

        self.cursor.execute(SQL_query)
        self.cursor.commit()
        print "Time taken in changing processed status:"
        print time.time()-startProcessed

    def fetchThresholdClaimLine(self):
        logging.info(' : '+__file__+' : '+'In fetchThresholdClaimLine() method.')
        SQL_query=(" select ThresholdValue from  ["+self.databasename+"].[dbo].["+self.thresholdCLMLineTable+"]"
                " where Name='"+str(self.thresholdClaimLineDescipInDB)+"'")


        self.cursor.execute(SQL_query)

        threshold=float(self.cursor.fetchone()[0])
        return threshold

    def putNoRevNecessaryTag(self):
        logging.info(' : '+__file__+' : '+'In putNoRevNecessaryTag() method.')
        startNorev=time.time()
        threshold=self.fetchThresholdClaimLine()
        SQL_query2=(" update ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"]"
                " set [REV_STAT]='No Review Necessary'"
                " where conf_score>="+str(threshold))


        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        print "Time taken in putting No review Necessary tag:"
        print time.time()-startNorev


    def checkPendingClaims(self):
        logging.info(' : '+__file__+' : '+'In checkPendingClaims() method.')
        startPending=time.time()
        SQL_query2=(" update ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"]"
                " set [PROCESSED_FLAG]='Unprocessed'"
                " from ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"] a, ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] as b"
                "  where (a.[REV_STAT]='Pending' and a.orig_desc=b.orig_desc and a.clm_id=b.clm_id and a.clm_ln_id=b.clm_ln_id and a.ReceivedDate=b.ReceivedDate)")
                # " or (a.[REV_STAT]='Rejected' and a.orig_desc=b.orig_desc and a.clm_id=b.clm_id and a.clm_ln_id=b.clm_ln_id)")


        SQL_querycheckPending=(" select count(distinct orig_desc) from ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"] where rev_stat='Pending'")
        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        self.cursor.execute(SQL_querycheckPending)
        noPendingClaims=float(self.cursor.fetchone()[0])
        self.cursor.commit()
        print "Time taken in changing processed flag for pending claims:"
        print time.time()-startPending
        return noPendingClaims

    def deletePendingClaims(self):
        logging.info(' : '+__file__+' : '+'In deletePendingClaims() method.')
        startdelete=time.time()
        SQL_query2=(" delete rep from ["+self.databasename+"].[dbo].["+self.classificationOutputTable+"] as rep"
                    " inner join ["+self.databasename+"].[dbo].["+self.claimLineHistoryTable+"] as hist on "
                "  (rep.rev_stat='Pending' and hist.CLM_ID=rep.CLM_ID and hist.CLM_LN_ID=rep.CLM_LN_ID and hist.ORIG_DESC=rep.ORIG_DESC and hist.ReceivedDate=rep.ReceivedDate)")



        self.cursor.execute(SQL_query2)
        self.cursor.commit()
        print "Time taken in deleting pending claims from output table:"
        print time.time()-startdelete


    def checkUnlearningRequest(self):
        logging.info(' : '+__file__+' : '+'In checkUnlearningRequest() method.')
        SQL_query=("select count(*) from ["+self.databasename+"].[dbo].["+self.unlearnMDTable+"] where processed_flag='Unprocessed'")

        # test=data.values.tolist()
        self.cursor.execute(SQL_query)
        no_of_request=int(self.cursor.fetchone()[0])
        return no_of_request

    def changeUnlearningStatus(self):
        logging.info(' : '+__file__+' : '+'In changeUnlearningStatus() method.')
        SQL_query=("update ["+self.databasename+"].[dbo].["+self.unlearnMDTable+"]  set ["+self.databasename+"].[dbo].["+self.unlearnMDTable+"].PROCESSED_FLAG='Processed' "
                    " from (select distinct UNLEARN_ID from ["+self.databasename+"].[dbo].["+self.unlearnDataTable+"]) as temp"
                    " where ["+self.databasename+"].[dbo].["+self.unlearnMDTable+"].[UNLEARN_ID]=temp.UNLEARN_ID")


        self.cursor.execute(SQL_query)
        self.cursor.commit()

    def updateUnlearningHistoryTable(self):
        logging.info(' : '+__file__+' : '+'In updateUnlearningHistoryTable() method.')
        SQL_query=("insert into ["+self.databasename+"].[dbo].["+self.unlearnHistoryTable+"]"
                        " ( [UNLEARN_DESC_ID], [ORIG_DESC] ,[CHRG_CLS],[CHRG_CLS_NEW],[REV_DT],[USER_ID],[UNLEARN_ID])"
                        " select [UNLEARN_DESC_ID], [ORIG_DESC] ,[CHRG_CLS],[CHRG_CLS_NEW],[REV_DT],[USER_ID],[UNLEARN_ID] from ["+self.databasename+"].[dbo].["+self.unlearnDataTable+"]")
        SQL_query1=("truncate table ["+self.databasename+"].[dbo].["+self.unlearnDataTable+"]")
        self.cursor.execute(SQL_query)
        self.cursor.commit()
        self.cursor.execute(SQL_query1)
        self.cursor.commit()

    def pullUnlearningRequests(self):
        logging.info(' : '+__file__+' : '+'In pullUnlearningRequests() method.')
        SQL_query=("select * from ["+self.databasename+"].[dbo].["+self.unlearnMDTable+"] where processed_flag='Unprocessed'")
        self.cursor.execute(SQL_query)
        requests=pd.DataFrame(self.cursor.fetchall())
        self.cursor.commit()
        return requests


    def doUnlearning(self):
        logging.info(' : '+__file__+' : '+'In doUnlearning() method.')
        startUnlearn=time.time()
        SQL_query=("update  train "
                 " set train.chrg_cls=a.chrg_cls_new"
                    " from ["+self.databasename+"].[dbo].["+self.trainingTable+"] train,"
                        " (select [UNLEARN_DESC_ID] ,data.[ORIG_DESC] ,[CHRG_CLS],[CHRG_CLS_NEW],[REV_DT],[USER_ID],[UNLEARN_ID]"
                        " FROM ["+self.databasename+"].[dbo].["+self.unlearnDataTable+"] as data,"
                        " (SELECT [ORIG_DESC],max([REV_DT]) as dat"
                        " FROM ["+self.databasename+"].[dbo].["+self.unlearnDataTable+"]"
                        " group by ORIG_DESC) as temp "
                        " where data.rev_dt=temp.dat and data.ORIG_DESC=temp.orig_desc ) as a"
                    " where train.orig_desc=a.orig_desc")

        self.cursor.execute(SQL_query)
        self.cursor.commit()
        print 'unlearning done'
        print "Time taken in unlearning:"
        print time.time()-startUnlearn




    def updateTrainingFromManualReview(self):
        logging.info(' : '+__file__+' : '+'In updateTrainingFromManualReview() method.')
        # handle NULL checks... #done
        from app.core import normalizer
        normal=normalizer.Normalizer()
        startMRUpdate=time.time()
        threshold=(self.fetchThresholdClaimLine())*100

        SQL_query1=(" select mr.orig_desc,mr.chrg_cls_new,mr.norm_desc_new,mr.ftr1_new,mr.ftr2_new,mr.ftr3_new,mr.ftr4_new "
                " from (SELECT   orig_desc, max(UpdatedDate) as dat"
                        " FROM ["+self.databasename+"].[dbo].["+self.mrDetailTable+"]"
                        " where rev_stat='Approved' group by orig_desc) as temp , ["+self.databasename+"].[dbo].["+self.mrDetailTable+"] as mr"
                " where (mr.orig_desc=temp.orig_desc and temp.dat=mr.UpdatedDate and mr.rev_stat='Approved' and mr.conf_score<"+str(threshold)+" ) "
                "       or (mr.orig_desc=temp.orig_desc and temp.dat=mr.UpdatedDate and mr.rev_stat='Approved' and mr.conf_score>="+str(threshold)+" and UpdatedBy not in ( ' ',''))")



        SQL_query2=(" merge ["+self.databasename+"].[dbo].["+self.trainingTable+"] as train"
                    " using ("
                        " select * from (values (?,?,?,?,?,?,?)) as temp(orig_desc,chrg_cls_new,norm_desc_new,ftr1_new,ftr2_new,ftr3_new,ftr4_new)) as manrev on (train.orig_desc=manrev.orig_desc)"
                    " when matched then "
                        " update set train.chrg_cls=manrev.chrg_cls_new, train.norm_desc=manrev.norm_desc_new,train.flag='ONLINE',train.create_dt=convert(date,getdate()),train.ftr1=manrev.ftr1_new,train.ftr2=manrev.ftr2_new,train.ftr3=manrev.ftr3_new,train.ftr4=manrev.ftr4_new"
                    " when not matched then "
                        " insert (orig_desc,chrg_cls,norm_desc,flag,create_dt,ftr1,ftr2,ftr3,ftr4)"
                        " values (manrev.orig_desc,manrev.chrg_cls_new,manrev.norm_desc_new,'ONLINE',convert(date,getdate()),manrev.ftr1_new,manrev.ftr2_new,manrev.ftr3_new,manrev.ftr4_new);")


        self.cursor.execute(SQL_query1)
        df=pd.DataFrame(self.cursor.fetchall(),columns=[appconfig.ORIGINAL_DESCRIPTION,appconfig.TARGET,appconfig.TEXT_FEATURE,'Feature 1','Feature 2','Feature 3','Feature 4'])
        df[appconfig.TEXT_FEATURE] = df[appconfig.TEXT_FEATURE].apply(lambda x: re.sub('[^A-Za-z0-9+%.>/&]',' ',x))
        df[appconfig.TEXT_FEATURE] = df[appconfig.TEXT_FEATURE].apply(lambda x: re.sub('[ ]{2,}',' ',x))
        df[appconfig.ORIGINAL_DESCRIPTION] = df[appconfig.ORIGINAL_DESCRIPTION].apply(lambda x: re.sub('[^A-Za-z0-9+%.>/&]',' ',x))
        df[appconfig.ORIGINAL_DESCRIPTION] = df[appconfig.ORIGINAL_DESCRIPTION].apply(lambda x: re.sub('[ ]{2,}',' ',x))
        # CorrectDesc = re.sub('[^A-Za-z0-9+%.>/&]',' ',CorrectDesc)
        # CorrectDesc = re.sub('[ ]{2,}',' ',CorrectDesc)
        # df[appconfig.TARGET] = df[appconfig.TARGET].apply(lambda x: ''.join([i if 32 < ord(i) < 126 else " " for i in x]))
        # handle NULL checks...return if df is empty #done
        if len(df) > 0:
            df=normal.addUnderscoreAbbreviationForMRdata(df,appconfig.TEXT_FEATURE)
            data=df.values.tolist()
            self.cursor.executemany(SQL_query2,data)
        self.cursor.commit()
        print "Time taken in updating training from MR:"
        print time.time()-startMRUpdate

    def checkNoMRRecords(self):
        logging.info(' : '+__file__+' : '+'In checkNoMRRecords() method.')
        SQL_query=(" select count(*)"
                        " from "
                            " (select distinct mra.orig_desc"
                            " from ["+self.databasename+"].[dbo].["+self.mrDetailTable+"] mra"
                            " where ("
	                                " select count(distinct chrg_cls_new)"
	                                " from ["+self.databasename+"].[dbo].["+self.mrDetailTable+"] mrb"
	                                " where mra.orig_desc=mrb.orig_desc"
                                    " )=1) as temp,["+self.databasename+"].[dbo].["+self.mrDetailTable+"] as mr"
                        " where temp.orig_desc=mr.orig_desc and mr.rev_stat='Approved'")

        self.cursor.execute(SQL_query)
        count=int(self.cursor.fetchone()[0])
        self.cursor.commit()
        return count

    def clearMRTable(self):
        logging.info(' : '+__file__+' : '+'In clearMRTable() method.')
        SQL_query=(" delete "
                        " from ["+self.databasename+"].[dbo].["+self.mrDetailTable+"]"
                        " where ["+self.databasename+"].[dbo].["+self.mrDetailTable+"].rev_stat='Approved'")

        self.cursor.execute(SQL_query)
        self.cursor.commit()

    def closeMetaMapServers(self):
        logging.info(' : '+__file__+' : '+'In closeMetaMapServers() method.')
        from win32com.client import GetObject
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

if __name__== "__main__":
    dbConnection=DatabaseConnections()
    dbConnection.autoClassifySet1()
    len2=dbConnection.pullSet2CompleteToCSV()
    dbConnection.closeConnection()
    # data=pd.read_csv("../release9/Medical Concepts for word2vec.csv" ,na_filter=False)
    # # data=data[['CLM_ID','CLM_LN_ID','ORIG_DESC','NORM_DESC','CHRG_CLS','PATIENT_NM','CLIENT_NM','PROV_NM','CLM_RCV_DT','EXP_DT','FTR1','FTR2','FTR3','FTR4','CONF_SCORE','REV_STAT']]
    # # data.insert(loc=3,column='Date',value=datetime.now())
    # print data
    # SQL_write="Driver={SQL Server};Server=WDVD1BRESQL01;Database=CE_WORK_Test;Trusted_Connection=True;"
    # write_connection = pypyodbc.connect(SQL_write)
    # cursor = write_connection.cursor()
    # SQL_query=("INSERT INTO MEDICAL_CONCEPTS_TAB "
    #              # "(CLM_ID,CLM_LN_ID,ORIG_DESC,NORM_DESC,CHRG_CLS,PATIENT_NM,CLIENT_NM,PROV_NM,CLM_RCV_DT,EXP_DT,FTR1,FTR2,FTR3,FTR4,CONF_SCORE,REV_STAT)"
    #              "VALUES (?,?)")#,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    # # SQL_query=("insert into tempchange (clm_id,change) values (?,?) ")
    #
    # test=data.values.tolist()
    # cursor.executemany(SQL_query, test)
    # cursor.commit()
    # cursor.close()
    # write_connection.close()
    # temp_uploadTestData()

