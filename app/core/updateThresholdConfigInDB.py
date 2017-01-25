import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from app.conf import dbconfig
import pypyodbc
import time



servername=dbconfig.SERVER_NAME
databasename=dbconfig.DATABASENAME
thresholdCLMLineTable=dbconfig.THRESHOLD_CONFIG
logFile = open('../../configlog.txt', 'a')
SQL_write="Driver={SQL Server};Server="+servername+";Database="+databasename+";Trusted_Connection=True;"
write_connection = pypyodbc.connect(SQL_write)
cursor = write_connection.cursor()
thresholdClaimLine=dbconfig.THRESHOLD_CLAIM_LINE
thresholdClaimLineIDinDB=dbconfig.THRESHOLD_CLAIM_LINE_ID_IN_DB
thresholdClaimLineDescripInDB=dbconfig.THRESHOLD_CLAIM_LINE_DESCRIP_IN_DB
# thresholdClaimID=dbconfig.THRESHOLD_CLAIMID
try:
    thresholdClaimLine=float(thresholdClaimLine)
    if thresholdClaimLine<0.0:
        thresholdClaimLine=0.0
        logFile.write(time.strftime("%H:%M:%S")+' :: Threshold value is lesser than 0.0 , value updated in database: 0.0 !\n')
    elif thresholdClaimLine>1.0:
        thresholdClaimLine=1.0
        logFile.write(time.strftime("%H:%M:%S")+' :: Threshold value is greater than 1.0 , value updated in database: 1.0 !\n')
    SQL_query=("update "+thresholdCLMLineTable+" "
                     " set ThresholdValue="+str(thresholdClaimLine)+" where Name='"+str(thresholdClaimLineDescripInDB)+"'")

    cursor.execute(SQL_query)
    cursor.commit()
    logFile.write(time.strftime("%H:%M:%S")+' ::Config value updated in database!!!\n')
    logFile.write('Threshold value: '+ str(thresholdClaimLine)+'\n')
    logFile.write('---------------------------------------------------\n')
except:
    logFile.write(time.strftime("%H:%M:%S")+' :: Threshold value can only be a real number(characters not allowed) in between 0.0 and 1.0 . !\n')

    logFile.write('---------------------------------------------------\n')
finally:
    logFile.close()
    cursor.close()
    write_connection.close()


