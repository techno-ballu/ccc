This Readme helps in explaining how to set values in the config files.

Config file types:	
 * Single valued .txt files. If there are 10 configurable parameters there will be 10 different config files. 
 * File name would recognize the type of configuration. 

When user needs to update the configuration values he needs to open the corresponding config .txt file and update the value accordingly. 

Ex: ApprovalThreshold.txt should contain the threshold value for claim lines (default value = 1.0)

Following is the current list of config files with default values:

1. ApprovalThreshold.txt - Enter value between 0.0 and 1.0
2. ClassificationCoverageThreshold.txt - Enter value between 0.0 and 1.0
3. FrequencyOfClassificationPipeline.txt
4. FrequencyOfDataPull.txt
5. FrequencyOfLearningPipeline.txt
6. SourceDatabase.txt
7. SourceDataServer.txt
8. SourceTable.txt
9. TargetDatabase.txt - Name of target database e.g. CE_Work_Test
10. TargetServer.txt - Name of target database server e.g. WDVD1BRESQL01
