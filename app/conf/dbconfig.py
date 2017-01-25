__author__ = 'Abzooba'

targetDBFile = open('../../config/TargetDatabase.txt', 'r')
targetServerFile = open('../../config/TargetServer.txt', 'r')
targetThresholdClaimLinefile=open('../../config/ApprovalThreshold.txt', 'r')
# targetThresholdClaimIDfile=open('../../config/ClassificationCoverageThreshold.txt', 'r')

SERVER_NAME=targetServerFile.read() # 'WDVD1BRESQL01' # TARGET SERVER
DATABASENAME= targetDBFile.read() # 'CE_Work_Test' # TARGET DB
THRESHOLD_CLAIM_LINE_ID_IN_DB=1000 # This is the id of claim line threshold value in ThresholdConfig table
THRESHOLD_CLAIM_LINE_DESCRIP_IN_DB='Claim Line' # This is the Name column of claim line threshold value in ThresholdConfig table
THRESHOLD_CLAIM_LINE=targetThresholdClaimLinefile.read()
# THRESHOLD_CLAIMID=targetThresholdClaimIDfile.read()
THRESHOLD_CLAIM_LINE=THRESHOLD_CLAIM_LINE.strip()
# THRESHOLD_CLAIMID=THRESHOLD_CLAIMID.strip()
SERVER_NAME=SERVER_NAME.strip()
DATABASENAME=DATABASENAME.strip()
TRAINING_TABLE='TRNG_TAB'
TRAINING_MD='TRNG_MD_TAB'
CLAIM_LN_HIST_TAB='CLM_LN_HIST_TAB'
DISTINCT_ORIG_DESC_TAB='ORIG_DESC_TAB'
CLASSIFICATION_TEMP_TAB='CLSF_TEMP_TAB'
CLASSF_MD_TAB='CLSF_METADATA_TAB'
CLSF_OUTPUT_TAB='CLM_LN_CLSF_REP_TAB'
UNLEARN_DATA_TAB='UNLEARN_DATA_TAB'
UNLEARN_MD_TAB='UNLEARN_METADATA_TAB'
MR_DETAIL_TAB='MR_DETAIL_TAB'
ABBR_CORPUS = 'ABBR_CORPUS'
SPELL_CORR_CORPUS = 'SPELL_CORR_CORPUS'
THRESHOLD_CONFIG='ThresholdConfig'
UNLEARN_HIS_TAB='UNLEARN_HIST_TAB'
MEDICAL_CONCEPTS_TABLE='MEDICAL_CONCEPTS_TAB'
CC_ALGORITHM_RESULT_TABLE = 'CC_ALGORITHM_RESULTS'
targetDBFile.close()
targetServerFile.close()
# targetThresholdClaimIDfile.close()
targetThresholdClaimLinefile.close()