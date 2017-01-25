from __future__ import division
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pandas as pd
import numpy as np
import pypyodbc
import re
from app.core import jaro_wrinkler_algo

def changeencode(data, cols):
    for col in cols:
        print 'cleaning', col
        data[col] = data[col].astype(str)
        data[col] = data[col].str.decode('iso-8859-1').str.encode('utf-8')
    return data

servername = 'WDVD1BRESQL01'
databasename = 'CE_WORK_TEST'
desc_table = 'CRR_LI_DESC'
desc_table_copy = 'CRR_LI_DESC_copy'
desc_table_temp = 'CRR_LI_DESC_temp'
unconfirmed_table = 'CRR_LI_UnConfirmed'
unconfirmed_table_copy = 'CRR_LI_UnConfirmed_copy'
classified_table = 'CRR_LI_Classified'
classified_table_copy = 'CRR_LI_Classified_copy'

def clean_data(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: re.sub('[^A-Za-z0-9+%.>/&]',' ',x))
        df[col] = df[col].apply(lambda x: re.sub('[ ]{2,}',' ',x))
        df[col] = df[col].apply(lambda x: x[3:] if x.startswith('HB ') or x.startswith('HC ') else x)
        print 'cleaning '+ col
    return df


SQL_write="Driver={SQL Server};Server="+servername+";Database="+databasename+";Trusted_Connection=True;"
write_connection = pypyodbc.connect(SQL_write)
cursor = write_connection.cursor()
# SQL_QUERY = ("SELECT descrip, proc_code, units, billed_amount," \
#             " rev_code, claim_id, facility_id, DESC_ID, NORM_DESC, CHRG_CLS"
#             "  FROM ["+databasename+"].[dbo].["+desc_table+"] as a "
# 	        "  INNER JOIN ["+databasename+"].[dbo].["+classified_table+"] as b"
# 	        " ON b.ORIG_DESC=a.descrip"
#             " ORDER BY descrip asc")

# SQL_QUERY_training_unmatched = (" insert into ["+databasename+"].[dbo].["+classified_table_copy+"] "
#                                 " SELECT  a.DESC_ID,a.ORIG_DESC, a.NORM_DESC, a.CHRG_CLS"
#             "  FROM ["+databasename+"].[dbo].["+classified_table+"] as a "
# 	        "  LEFT JOIN ["+databasename+"].[dbo].["+desc_table+"] as b"
# 	        " ON a.ORIG_DESC=b.descrip where b.descrip is null "
#             " ORDER BY descrip asc")


# SQL_QUERY_fetchUnmatched = ("SELECT top 20 *"
#             "  FROM ["+databasename+"].[dbo].["+classified_table_copy+"] ")

SQL_QUERY_fetchTraining = ("SELECT *"
            "  FROM ["+databasename+"].[dbo].["+classified_table+"] ")

SQL_QUERY_fetchTest = ("SELECT *"
            "  FROM ["+databasename+"].[dbo].["+unconfirmed_table+"] ")

SQL_QUERY_fetchStructured = ("SELECT distinct descrip"
            "  FROM ["+databasename+"].[dbo].["+desc_table+"] ")
SQL_QUERY_upload_cleaned_str = ("insert into "
            "  ["+databasename+"].[dbo].["+desc_table_temp+"] values (?,?)")
SQL_QUERY_upload_cleaned_training = ("insert into "
            "  ["+databasename+"].[dbo].["+classified_table_copy+"] values (?,?,?,?,?)")
SQL_QUERY_upload_cleaned_test = ("insert into "
            "  ["+databasename+"].[dbo].["+unconfirmed_table_copy+"] values (?,?,?,?,?,?,?)")
SQL_QUERY_join_desc_table = ("insert into "
            "  ["+databasename+"].[dbo].["+desc_table_copy+"] "
            " select a.line_item_created, a.service_date, a.proc_code, a.descrip,a.units,a.billed_amount,a.rev_code, a.claim_id"
            " ,a.facility_id, b.cleaned_descrip from ["+databasename+"].[dbo].["+desc_table+"] as a"
            " left join ["+databasename+"].[dbo].["+desc_table_temp+"] as b on a.descrip = b.descrip")


cursor.execute(SQL_QUERY_fetchTraining)
print 'fetched training data'
df_training = pd.DataFrame(cursor.fetchall(),columns=['DESC_ID','ORIG_DESC','NORM_DESC','CHRG_CLS'])
df_training['cleaned_orig_desc'] = df_training['ORIG_DESC']
df_training = clean_data(df_training,['cleaned_orig_desc'])
df_training = df_training.values.tolist()
cursor.executemany(SQL_QUERY_upload_cleaned_training, df_training)
print 'uploaded cleaned training data'
cursor.commit()

cursor.execute(SQL_QUERY_fetchTest)
print 'fetched test data'
df_test = pd.DataFrame(cursor.fetchall(),columns=['clm_id', 'clm_ln_id','ORIG_DESC','NORM_DESC','CHRG_CLS','PROV_NM'])
df_test['cleaned_orig_desc'] = df_test['ORIG_DESC']
df_test = clean_data(df_test,['cleaned_orig_desc'])
df_test = df_test.values.tolist()
cursor.executemany(SQL_QUERY_upload_cleaned_test, df_test)
print 'uploaded cleaned test data'
cursor.commit()


cursor.execute(SQL_QUERY_fetchStructured)
print 'fetched structured data'
df_allStructured = pd.DataFrame(cursor.fetchall(),columns=['descrip'])
df_allStructured['descrip'] = df_allStructured['descrip'].astype(str)
df_allStructured['cleaned_descrip'] = df_allStructured['descrip']
df_allStructured = clean_data(df_allStructured,['cleaned_descrip'])
df_allStructured = df_allStructured.values.tolist()
cursor.executemany(SQL_QUERY_upload_cleaned_str, df_allStructured)
print 'uploaded cleaned unique structured data'
cursor.commit()

cursor.execute(SQL_QUERY_join_desc_table)
print 'uploaded complete structured data'
cursor.commit()
cursor.close()
write_connection.close()


