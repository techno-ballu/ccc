from __future__ import division
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pandas as pd
import plots

df_structured = pd.read_csv('../release10/df_train_struc_joined.csv', encoding='utf_8', sep='\t')
print df_structured.columns
print df_structured.shape
print '# of distinct original descriptions = ', len(df_structured['descrip'].unique())
print '# of distinct charge classes = ', len(df_structured['CHRG_CLS'].unique())
# print df_structured['proc_code'].value_counts();print

print '# of distinct revenue codes = ', len(df_structured['Rev_Code_Name'].unique())
# print '# of distinct procedure codes = ', len(df_structured['proc_code'].unique())

# plots.plotHistogram(df_structured['norm_billed_amount'].values)
# plots.plotHistogram(df_structured['units'].values)
max_billed = df_structured['norm_billed_amount'].max()
bins = [-10, 0, 4, 17, 41, 61, 92, 287, 641, 997, 2496, 9320, max_billed]
# group_names = ['Zero $ value', 'Low $ value', 'Okay $ value', 'High $ value']
df_structured['billed_categories'] = pd.cut(df_structured['norm_billed_amount'], bins) # , labels=group_names
print df_structured['billed_categories'].value_counts()

max_units = df_structured['units'].max()
bins = [0, 1.00, max_units]
group_names = ['Units <= 1', 'Units > 1']
df_structured['units_categories'] = pd.cut(df_structured['units'], bins, labels=group_names) #
print df_structured['units_categories'].value_counts()

# # Let's bin some of the rev codes!
# df_structured['Rev_Code_Name'] = 'Others'
# df_structured.loc[(df_structured['rev_code'] >= 110) & (df_structured['rev_code'] <= 179),
#                   'Rev_Code_Name'] = 'Room & Board'
# df_structured.loc[(df_structured['rev_code'] >= 200) & (df_structured['rev_code'] <= 209),
#                   'Rev_Code_Name'] = 'Intensive Care'
# df_structured.loc[(df_structured['rev_code'] >= 210) & (df_structured['rev_code'] <= 219),
#                   'Rev_Code_Name'] = 'Coronary Care'
# df_structured.loc[(df_structured['rev_code'] >= 220) & (df_structured['rev_code'] <= 229),
#                   'Rev_Code_Name'] = 'Special Charges'
# df_structured.loc[(df_structured['rev_code'] >= 230) & (df_structured['rev_code'] <= 239),
#                   'Rev_Code_Name'] = 'Nursing'
# df_structured.loc[(df_structured['rev_code'] >= 240) & (df_structured['rev_code'] <= 249),
#                   'Rev_Code_Name'] = 'Ancillary Services'
# df_structured.loc[(df_structured['rev_code'] >= 250) & (df_structured['rev_code'] <= 259),
#                   'Rev_Code_Name'] = 'Pharmacy'
# df_structured.loc[(df_structured['rev_code'] >= 260) & (df_structured['rev_code'] <= 269),
#                   'Rev_Code_Name'] = 'IV Therapy'
# df_structured.loc[(df_structured['rev_code'] >= 270) & (df_structured['rev_code'] <= 279),
#                   'Rev_Code_Name'] = 'Implant/Supply'
# df_structured.loc[(df_structured['rev_code'] >= 280) & (df_structured['rev_code'] <= 289),
#                   'Rev_Code_Name'] = 'Oncology'
# df_structured.loc[(df_structured['rev_code'] >= 290) & (df_structured['rev_code'] <= 299),
#                   'Rev_Code_Name'] = 'Medical Equipment'
# df_structured.loc[(df_structured['rev_code'] >= 300) & (df_structured['rev_code'] <= 319),
#                   'Rev_Code_Name'] = 'Laboratory'
# df_structured.loc[(df_structured['rev_code'] >= 320) & (df_structured['rev_code'] <= 339),
#                   'Rev_Code_Name'] = 'Radiology'
# df_structured.loc[(df_structured['rev_code'] >= 340) & (df_structured['rev_code'] <= 349),
#                   'Rev_Code_Name'] = 'Nuclear Medicine'
# df_structured.loc[(df_structured['rev_code'] >= 350) & (df_structured['rev_code'] <= 359),
#                   'Rev_Code_Name'] = 'CT Scan'
# df_structured.loc[(df_structured['rev_code'] >= 360) & (df_structured['rev_code'] <= 369),
#                   'Rev_Code_Name'] = 'OR Services'
# df_structured.loc[(df_structured['rev_code'] >= 370) & (df_structured['rev_code'] <= 379),
#                   'Rev_Code_Name'] = 'Anesthesia'
# df_structured.loc[(df_structured['rev_code'] >= 380) & (df_structured['rev_code'] <= 399),
#                   'Rev_Code_Name'] = 'Blood'
# df_structured.loc[(df_structured['rev_code'] >= 400) & (df_structured['rev_code'] <= 409),
#                   'Rev_Code_Name'] = 'Other Imaging Services'
# df_structured.loc[(df_structured['rev_code'] >= 410) & (df_structured['rev_code'] <= 419),
#                   'Rev_Code_Name'] = 'Respiratory Services'
# df_structured.loc[(df_structured['rev_code'] >= 420) & (df_structured['rev_code'] <= 449),
#                   'Rev_Code_Name'] = 'PT/OT/Speech'
# df_structured.loc[(df_structured['rev_code'] >= 450) & (df_structured['rev_code'] <= 459),
#                   'Rev_Code_Name'] = 'Emergency Room'
# df_structured.loc[(df_structured['rev_code'] >= 460) & (df_structured['rev_code'] <= 469),
#                   'Rev_Code_Name'] = 'Pulmonary Function'
# df_structured.loc[(df_structured['rev_code'] >= 470) & (df_structured['rev_code'] <= 479),
#                   'Rev_Code_Name'] = 'Audiology'
# df_structured.loc[(df_structured['rev_code'] >= 480) & (df_structured['rev_code'] <= 489),
#                   'Rev_Code_Name'] = 'Cardiology'
# df_structured.loc[(df_structured['rev_code'] >= 490) & (df_structured['rev_code'] <= 499),
#                   'Rev_Code_Name'] = 'Ambulatory Surgical'
# df_structured.loc[(df_structured['rev_code'] >= 500) & (df_structured['rev_code'] <= 509),
#                   'Rev_Code_Name'] = 'Outpatient Services'
# df_structured.loc[(df_structured['rev_code'] >= 510) & (df_structured['rev_code'] <= 529),
#                   'Rev_Code_Name'] = 'Clinic'
# df_structured.loc[(df_structured['rev_code'] >= 530) & (df_structured['rev_code'] <= 539),
#                   'Rev_Code_Name'] = 'Osteopathic Services'
# df_structured.loc[(df_structured['rev_code'] >= 540) & (df_structured['rev_code'] <= 549),
#                   'Rev_Code_Name'] = 'Ambulance'
# df_structured.loc[(df_structured['rev_code'] >= 550) & (df_structured['rev_code'] <= 569),
#                   'Rev_Code_Name'] = 'Nursing'
# df_structured.loc[(df_structured['rev_code'] >= 570) & (df_structured['rev_code'] <= 609),
#                   'Rev_Code_Name'] = 'Home Health'
# df_structured.loc[(df_structured['rev_code'] >= 610) & (df_structured['rev_code'] <= 619),
#                   'Rev_Code_Name'] = 'MRI'
# df_structured.loc[(df_structured['rev_code'] >= 620) & (df_structured['rev_code'] <= 629),
#                   'Rev_Code_Name'] = 'Implant/Supply'
# df_structured.loc[(df_structured['rev_code'] >= 630) & (df_structured['rev_code'] <= 639),
#                   'Rev_Code_Name'] = 'Drugs Requiring Specific Identification'
# df_structured.loc[(df_structured['rev_code'] >= 640) & (df_structured['rev_code'] <= 649),
#                   'Rev_Code_Name'] = 'Home IV Therapy Services'
# df_structured.loc[(df_structured['rev_code'] >= 650) & (df_structured['rev_code'] <= 659),
#                   'Rev_Code_Name'] = 'Hospice Services'
# df_structured.loc[(df_structured['rev_code'] >= 660) & (df_structured['rev_code'] <= 669),
#                   'Rev_Code_Name'] = 'Respite Care'
# df_structured.loc[(df_structured['rev_code'] >= 670) & (df_structured['rev_code'] <= 679),
#                   'Rev_Code_Name'] = 'Outpatient Special Residence Charges'
# df_structured.loc[(df_structured['rev_code'] >= 700) & (df_structured['rev_code'] <= 729),
#                   'Rev_Code_Name'] = 'Cast/Recovery/Labor Room'
# df_structured.loc[(df_structured['rev_code'] >= 730) & (df_structured['rev_code'] <= 749),
#                   'Rev_Code_Name'] = 'EKG/ECG/EEG'
# df_structured.loc[(df_structured['rev_code'] >= 750) & (df_structured['rev_code'] <= 759),
#                   'Rev_Code_Name'] = 'Gastro-Intestinal Services'
# df_structured.loc[(df_structured['rev_code'] >= 760) & (df_structured['rev_code'] <= 769),
#                   'Rev_Code_Name'] = 'Treatment/Observation Room'
# df_structured.loc[(df_structured['rev_code'] >= 770) & (df_structured['rev_code'] <= 809),
#                   'Rev_Code_Name'] = 'Preventive Care Services'
# df_structured.loc[(df_structured['rev_code'] >= 810) & (df_structured['rev_code'] <= 819),
#                   'Rev_Code_Name'] = 'Organ Acquisition'
# df_structured.loc[(df_structured['rev_code'] >= 820) & (df_structured['rev_code'] <= 859),
#                   'Rev_Code_Name'] = 'Outpatient or Home'
# df_structured.loc[(df_structured['rev_code'] >= 900) & (df_structured['rev_code'] <= 919),
#                   'Rev_Code_Name'] = 'Psychia/Psycho'
# df_structured.loc[(df_structured['rev_code'] >= 920) & (df_structured['rev_code'] <= 929),
#                   'Rev_Code_Name'] = 'Other Diagnostic Services'
# df_structured.loc[(df_structured['rev_code'] >= 940) & (df_structured['rev_code'] <= 959),
#                   'Rev_Code_Name'] = 'Other Therapeutic Services'
# df_structured.loc[(df_structured['rev_code'] >= 960) & (df_structured['rev_code'] <= 989),
#                   'Rev_Code_Name'] = 'Prof Fees'
# df_structured.loc[(df_structured['rev_code'] >= 990) & (df_structured['rev_code'] <= 999),
#                   'Rev_Code_Name'] = 'Patient Conv Items'
#
# print df_structured['Rev_Code_Name'].value_counts();print
# # print df_structured['rev_code'].value_counts();print

df_grouped = plots.groupByIndexColumn(df_structured, 'CHRG_CLS', 'Rev_Code_Name', 'descrip')
plots.plotHeatMap(df_grouped, rotate=True, cmap='Blues')

# df_grouped = plots.groupByIndexColumn(df_structured, 'CHRG_CLS', 'proc_code', 'descrip')
# plots.plotHeatMap(df_grouped, rotate=True, cmap='Blues')

df_grouped = plots.groupByIndexColumn(df_structured, 'CHRG_CLS', 'billed_categories', 'descrip')
plots.plotHeatMap(df_grouped, annotate = True, rotate=True, cmap='Blues')

df_grouped = plots.groupByIndexColumn(df_structured, 'CHRG_CLS', 'units_categories', 'descrip')
plots.plotHeatMap(df_grouped, annotate = True, rotate=False, cmap='Blues')

# gps = df_structured['proc_code'].value_counts().index
# groups = df_structured.groupby('proc_code')
# rows = []
# for gp in gps:
#     # if gp == u'-100.0':
#     #     continue
#     group = groups.get_group(gp)
#     # print gp, len(group), ', '.join(group['CHRG_CLS'].unique()), ', '.join(group['Rev_Code_Name'].unique());print
#     rows.append([gp, len(group), ', '.join(group['CHRG_CLS'].unique()), ', '.join(group['Rev_Code_Name'].unique())])
#                  # ', '.join(group['rev_code'].unique())
# df_proc_codes = pd.DataFrame(rows, columns=['Procedure Code', '# of desciptions', 'Unique charge classes',
#                                             'Unique rev codes names']) # , 'Unique rev codes'

# gps = df_structured['descrip'].value_counts().index
groups = df_structured.groupby(['Rev_Code_Name', 'CHRG_CLS'])
rows = []
for name, group in groups:
    rows.append([name[0], name[1], len(group)])
                 # ', '.join(group['rev_code'].unique()), len(group['Rev_Code_Name'].unique()), len(group['CHRG_CLS'].unique()),
                 # ', '.join(group['CHRG_CLS'].unique()), ', '.join(group['Rev_Code_Name'].unique())
df_rev_codes = pd.DataFrame(rows, columns=['Rev Code Grp', 'CHRG_CLS', '# of descs']) #, '# of rev code grps', '# of charge classes',
                                            # 'Unique charge classes', 'Unique rev codes grps'


# df_proc_codes.to_csv('../release10/df_proc_codes.csv', encoding='latin1', index=False, sep=',')
df_rev_codes.to_csv('../release10/df_rev_codes.csv', encoding='latin1', index=False, sep=',')
df_structured.to_csv('../release10/df_train_struc_joined_rev_names.csv', encoding='latin1', index=False, sep=',')