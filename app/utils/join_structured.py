import pandas as pd
import numpy as np
import pypyodbc

def changeencode(data, cols):
    for col in cols:
        print 'cleaning', col
        data[col] = data[col].astype(str)
        data[col] = data[col].str.decode('iso-8859-1').str.encode('utf-8')
    return data

def join_structured(df_structured):
    df_structured=df_structured.replace('"','',regex=True)
    df_structured=df_structured.replace('\t','',regex=True)

    columns_to_clean = ['descrip', 'proc_code',  'claim_id', 'facility_id',  'NORM_DESC', 'CHRG_CLS'] #'rev_code',
    # df_structured = changeencode(df_structured, ['descrip', 'NORM_DESC'])
    df_structured = changeencode(df_structured, columns_to_clean)

    # df_train = pd.read_csv('../../data/training_11381.csv')
    # df_structured = pd.read_csv('../../data/structured_details_joined_raw.csv', encoding='latin1', sep=',')
    # df_structured.rename(columns={df_structured.columns[0]:'descrip'}, inplace=True)
    # print '='*30, 'Training Details:', '='*30
    # print len(df_train)
    # print df_train.columns
    print '='*30, 'Structured Data Analysis:', '='*30
    print df_structured.shape
    print len(df_structured)
    print len(df_structured['cleaned_descrip'].unique())
    print df_structured.columns

    # Clean the


    df_structured = df_structured[['descrip','cleaned_descrip', 'rev_code', 'proc_code', 'units', 'billed_amount', 'claim_id', 'NORM_DESC',
                                   'CHRG_CLS']]
    df_structured = df_structured.drop(df_structured[(df_structured['billed_amount'] > 0)
                                                     & (df_structured['units'] == 0)].index)
    df_structured = df_structured.loc[df_structured['billed_amount'] >= 0]
    df_structured = df_structured.loc[df_structured['units'] > 0]
    df_structured = df_structured.loc[(df_structured['rev_code'] > 99) & (df_structured['rev_code'] <= 999)]

    print len(df_structured)
    print len(df_structured['cleaned_descrip'].unique())

    print 'min, max of units:'
    print df_structured['units'].min(), df_structured['units'].max()
    print 'min, max of billed_amount:'
    print df_structured['billed_amount'].min(), df_structured['billed_amount'].max()
    df_structured = df_structured.fillna('-100.0')
    df_structured['norm_billed_amount'] = -100.0
    # df_structured.loc[df_structured['units'] > 0, 'norm_billed_amount'] = df_structured['billed_amount']/df_structured['units']
    df_structured['norm_billed_amount'] = df_structured['billed_amount']/df_structured['units']

    # Let's bin some of the rev codes!
    df_structured['Rev_Code_Name'] = 'Others'
    df_structured.loc[(df_structured['rev_code'] >= 110) & (df_structured['rev_code'] <= 179),
                      'Rev_Code_Name'] = 'Room & Board'
    df_structured.loc[(df_structured['rev_code'] >= 200) & (df_structured['rev_code'] <= 209),
                      'Rev_Code_Name'] = 'Intensive Care'
    df_structured.loc[(df_structured['rev_code'] >= 210) & (df_structured['rev_code'] <= 219),
                      'Rev_Code_Name'] = 'Coronary Care'
    df_structured.loc[(df_structured['rev_code'] >= 220) & (df_structured['rev_code'] <= 229),
                      'Rev_Code_Name'] = 'Special Charges'
    df_structured.loc[(df_structured['rev_code'] >= 230) & (df_structured['rev_code'] <= 239),
                      'Rev_Code_Name'] = 'Nursing'
    df_structured.loc[(df_structured['rev_code'] >= 240) & (df_structured['rev_code'] <= 249),
                      'Rev_Code_Name'] = 'Ancillary Services'
    df_structured.loc[(df_structured['rev_code'] >= 250) & (df_structured['rev_code'] <= 259),
                      'Rev_Code_Name'] = 'Pharmacy'
    df_structured.loc[(df_structured['rev_code'] >= 260) & (df_structured['rev_code'] <= 269),
                      'Rev_Code_Name'] = 'IV Therapy'
    df_structured.loc[(df_structured['rev_code'] >= 270) & (df_structured['rev_code'] <= 279),
                      'Rev_Code_Name'] = 'Implant/Supply'
    df_structured.loc[(df_structured['rev_code'] >= 280) & (df_structured['rev_code'] <= 289),
                      'Rev_Code_Name'] = 'Oncology'
    df_structured.loc[(df_structured['rev_code'] >= 290) & (df_structured['rev_code'] <= 299),
                      'Rev_Code_Name'] = 'Medical Equipment'
    df_structured.loc[(df_structured['rev_code'] >= 300) & (df_structured['rev_code'] <= 319),
                      'Rev_Code_Name'] = 'Laboratory'
    df_structured.loc[(df_structured['rev_code'] >= 320) & (df_structured['rev_code'] <= 339),
                      'Rev_Code_Name'] = 'Radiology'
    df_structured.loc[(df_structured['rev_code'] >= 340) & (df_structured['rev_code'] <= 349),
                      'Rev_Code_Name'] = 'Nuclear Medicine'
    df_structured.loc[(df_structured['rev_code'] >= 350) & (df_structured['rev_code'] <= 359),
                      'Rev_Code_Name'] = 'CT Scan'
    df_structured.loc[(df_structured['rev_code'] >= 360) & (df_structured['rev_code'] <= 369),
                      'Rev_Code_Name'] = 'OR Services'
    df_structured.loc[(df_structured['rev_code'] >= 370) & (df_structured['rev_code'] <= 379),
                      'Rev_Code_Name'] = 'Anesthesia'
    df_structured.loc[(df_structured['rev_code'] >= 380) & (df_structured['rev_code'] <= 399),
                      'Rev_Code_Name'] = 'Blood'
    df_structured.loc[(df_structured['rev_code'] >= 400) & (df_structured['rev_code'] <= 409),
                      'Rev_Code_Name'] = 'Other Imaging Services'
    df_structured.loc[(df_structured['rev_code'] >= 410) & (df_structured['rev_code'] <= 419),
                      'Rev_Code_Name'] = 'Respiratory Services'
    df_structured.loc[(df_structured['rev_code'] >= 420) & (df_structured['rev_code'] <= 449),
                      'Rev_Code_Name'] = 'PT/OT/Speech'
    df_structured.loc[(df_structured['rev_code'] >= 450) & (df_structured['rev_code'] <= 459),
                      'Rev_Code_Name'] = 'Emergency Room'
    df_structured.loc[(df_structured['rev_code'] >= 460) & (df_structured['rev_code'] <= 469),
                      'Rev_Code_Name'] = 'Pulmonary Function'
    df_structured.loc[(df_structured['rev_code'] >= 470) & (df_structured['rev_code'] <= 479),
                      'Rev_Code_Name'] = 'Audiology'
    df_structured.loc[(df_structured['rev_code'] >= 480) & (df_structured['rev_code'] <= 489),
                      'Rev_Code_Name'] = 'Cardiology'
    df_structured.loc[(df_structured['rev_code'] >= 490) & (df_structured['rev_code'] <= 499),
                      'Rev_Code_Name'] = 'Ambulatory Surgical'
    df_structured.loc[(df_structured['rev_code'] >= 500) & (df_structured['rev_code'] <= 509),
                      'Rev_Code_Name'] = 'Outpatient Services'
    df_structured.loc[(df_structured['rev_code'] >= 510) & (df_structured['rev_code'] <= 529),
                      'Rev_Code_Name'] = 'Clinic'
    df_structured.loc[(df_structured['rev_code'] >= 530) & (df_structured['rev_code'] <= 539),
                      'Rev_Code_Name'] = 'Osteopathic Services'
    df_structured.loc[(df_structured['rev_code'] >= 540) & (df_structured['rev_code'] <= 549),
                      'Rev_Code_Name'] = 'Ambulance'
    df_structured.loc[(df_structured['rev_code'] >= 550) & (df_structured['rev_code'] <= 569),
                      'Rev_Code_Name'] = 'Nursing'
    df_structured.loc[(df_structured['rev_code'] >= 570) & (df_structured['rev_code'] <= 609),
                      'Rev_Code_Name'] = 'Home Health'
    df_structured.loc[(df_structured['rev_code'] >= 610) & (df_structured['rev_code'] <= 619),
                      'Rev_Code_Name'] = 'MRI'
    df_structured.loc[(df_structured['rev_code'] >= 620) & (df_structured['rev_code'] <= 629),
                      'Rev_Code_Name'] = 'Implant/Supply'
    df_structured.loc[(df_structured['rev_code'] >= 630) & (df_structured['rev_code'] <= 639),
                      'Rev_Code_Name'] = 'Drugs Requiring Specific Identification'
    df_structured.loc[(df_structured['rev_code'] >= 640) & (df_structured['rev_code'] <= 649),
                      'Rev_Code_Name'] = 'Home IV Therapy Services'
    df_structured.loc[(df_structured['rev_code'] >= 650) & (df_structured['rev_code'] <= 659),
                      'Rev_Code_Name'] = 'Hospice Services'
    df_structured.loc[(df_structured['rev_code'] >= 660) & (df_structured['rev_code'] <= 669),
                      'Rev_Code_Name'] = 'Respite Care'
    df_structured.loc[(df_structured['rev_code'] >= 670) & (df_structured['rev_code'] <= 679),
                      'Rev_Code_Name'] = 'Outpatient Special Residence Charges'
    df_structured.loc[(df_structured['rev_code'] >= 700) & (df_structured['rev_code'] <= 729),
                      'Rev_Code_Name'] = 'Cast/Recovery/Labor Room'
    df_structured.loc[(df_structured['rev_code'] >= 730) & (df_structured['rev_code'] <= 749),
                      'Rev_Code_Name'] = 'EKG/ECG/EEG'
    df_structured.loc[(df_structured['rev_code'] >= 750) & (df_structured['rev_code'] <= 759),
                      'Rev_Code_Name'] = 'Gastro-Intestinal Services'
    df_structured.loc[(df_structured['rev_code'] >= 760) & (df_structured['rev_code'] <= 769),
                      'Rev_Code_Name'] = 'Treatment/Observation Room'
    df_structured.loc[(df_structured['rev_code'] >= 770) & (df_structured['rev_code'] <= 809),
                      'Rev_Code_Name'] = 'Preventive Care Services'
    df_structured.loc[(df_structured['rev_code'] >= 810) & (df_structured['rev_code'] <= 819),
                      'Rev_Code_Name'] = 'Organ Acquisition'
    df_structured.loc[(df_structured['rev_code'] >= 820) & (df_structured['rev_code'] <= 859),
                      'Rev_Code_Name'] = 'Outpatient or Home'
    df_structured.loc[(df_structured['rev_code'] >= 900) & (df_structured['rev_code'] <= 919),
                      'Rev_Code_Name'] = 'Psychia/Psycho'
    df_structured.loc[(df_structured['rev_code'] >= 920) & (df_structured['rev_code'] <= 929),
                      'Rev_Code_Name'] = 'Other Diagnostic Services'
    df_structured.loc[(df_structured['rev_code'] >= 940) & (df_structured['rev_code'] <= 959),
                      'Rev_Code_Name'] = 'Other Therapeutic Services'
    df_structured.loc[(df_structured['rev_code'] >= 960) & (df_structured['rev_code'] <= 989),
                      'Rev_Code_Name'] = 'Prof Fees'
    df_structured.loc[(df_structured['rev_code'] >= 990) & (df_structured['rev_code'] <= 999),
                      'Rev_Code_Name'] = 'Patient Conv Items'




    df_structured.loc[(df_structured['rev_code'] == 258) , 'Rev_Code_Name'] = 'IV Solutions'
    df_structured.loc[(df_structured['rev_code'] == 278) , 'Rev_Code_Name'] = 'Other Implants'
    df_structured.loc[(df_structured['rev_code'] == 636) , 'Rev_Code_Name'] = 'Drugs Requiring Detailed Coding'
    df_structured.loc[(df_structured['rev_code'] == 331) | (df_structured['rev_code'] == 332) | (df_structured['rev_code'] == 974)
                        , 'Rev_Code_Name'] = 'Pharmacy'
    df_structured.loc[(df_structured['rev_code'] >= 740) & (df_structured['rev_code'] <= 749),
                      'Rev_Code_Name'] = 'Surgery/Procedure'
    df_structured.loc[(df_structured['rev_code'] == 361) | (df_structured['rev_code'] == 362) | (df_structured['rev_code'] == 367)
                       | (df_structured['rev_code'] == 922) , 'Rev_Code_Name'] = 'Surgery/Procedure'
    df_structured.loc[(df_structured['rev_code'] == 972) | (df_structured['rev_code'] == 973)
                    | (df_structured['rev_code'] == 921), 'Rev_Code_Name'] = 'Radiology'
    df_structured.loc[(df_structured['rev_code'] == 391) , 'Rev_Code_Name'] = 'Blood Administration'

    print df_structured['Rev_Code_Name'].value_counts();print
    # print df_structured['rev_code'].value_counts();print

    groups = df_structured.groupby(['cleaned_descrip', 'Rev_Code_Name']) # 'rev_code', 'proc_code'
    print 'no of groups = ', len(groups)

    rows = []
    for name, group in groups:
        row = []
        chrg_cls = group['CHRG_CLS'].unique()
        norm_descs = group['NORM_DESC'].unique()
        descrip = group['descrip'].unique()
        # if len(chrg_cls) > 1:
            # print name[0], ', '.join(chrg_cls)
        row.extend([descrip[0],name[0], name[1], group['units'].median(), group['billed_amount'].median(),
                    group['norm_billed_amount'].median(), norm_descs[0], chrg_cls[0]])
        rows.append(row)

    df_aggregated = pd.DataFrame(rows, columns=['descrip','cleaned_descrip', 'Rev_Code_Name', 'units', 'billed_amount',
                                                'norm_billed_amount', 'NORM_DESC', 'CHRG_CLS']) # , 'proc_code'
    # print df_aggregated
    # , 'claim_id' , 'norm_billed_amount', 'facility_id', 'DESC_ID'
    # df_joined = pd.merge(df_aggregated, df_train, how='left', left_on='descrip', right_on='ORIG_DESC',
    #                      sort=True, suffixes=('_struc', '_train'), copy=True, indicator=False)
    print df_aggregated.shape
    print  df_aggregated.columns
    return df_aggregated



servername = 'WDVD1BRESQL01'
databasename = 'CE_WORK_TEST'
desc_table = 'CRR_LI_DESC'
desc_table_copy = 'CRR_LI_DESC_copy'
desc_table_temp = 'CRR_LI_DESC_temp'
unconfirmed_table = 'CRR_LI_UnConfirmed'
unconfirmed_table_copy = 'CRR_LI_UnConfirmed_copy'
classified_table = 'CRR_LI_Classified'
classified_table_copy = 'CRR_LI_Classified_copy'

SQL_write="Driver={SQL Server};Server="+servername+";Database="+databasename+";Trusted_Connection=True;"
write_connection = pypyodbc.connect(SQL_write)
cursor = write_connection.cursor()
SQL_QUERY_train = ("SELECT descrip , a.cleaned_descrip, proc_code, units, billed_amount," \
            " rev_code, claim_id, facility_id, DESC_ID, NORM_DESC, CHRG_CLS"
            "  FROM ["+databasename+"].[dbo].["+desc_table_copy+"] as a "
	        "  INNER JOIN ["+databasename+"].[dbo].["+classified_table_copy+"] as b"
	        " ON b.cleaned_ORIG_DESC=a.cleaned_descrip"
            " ORDER BY descrip asc")

SQL_QUERY_test = ("SELECT descrip , a.cleaned_descrip, proc_code, units, billed_amount," \
            " rev_code, claim_id, facility_id,PROV_NM, NORM_DESC, CHRG_CLS"
            "  FROM ["+databasename+"].[dbo].["+desc_table_copy+"] as a "
	        "  INNER JOIN ["+databasename+"].[dbo].["+unconfirmed_table_copy+"] as b"
	        " ON b.cleaned_ORIG_DESC=a.cleaned_descrip"
            " ORDER BY descrip asc")

cursor.execute(SQL_QUERY_train)
df_structured_train = pd.DataFrame(cursor.fetchall(),columns=['descrip','cleaned_descrip', 'proc_code', 'units', 'billed_amount', \
            'rev_code', 'claim_id', 'facility_id', 'DESC_ID', 'NORM_DESC', 'CHRG_CLS'])

cursor.execute(SQL_QUERY_test)
df_structured_test = pd.DataFrame(cursor.fetchall(),columns=['descrip','cleaned_descrip', 'proc_code', 'units', 'billed_amount', \
            'rev_code', 'claim_id', 'facility_id', 'PROV_NM', 'NORM_DESC', 'CHRG_CLS'])

cursor.close()
write_connection.close()

aggregated_train = join_structured(df_structured_train)
aggregated_train.to_csv('../../data/df_train_struc_joined.tsv', encoding='utf_8', index=False, sep='\t')
aggregated_test = join_structured(df_structured_test)
aggregated_test.to_csv('../../data/df_test_struc_joined.tsv', encoding='utf_8', index=False, sep='\t')

