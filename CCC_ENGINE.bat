call "C:\BusEngineRep\charge_class_classification\ssis\Batch files\equian_source_data_load.bat"
call "C:\BusEngineRep\charge_class_classification\machine_learning\runTrainingPipeline.bat"
call "C:\BusEngineRep\charge_class_classification\machine_learning\updateThresholdConfigInDB.bat"
call "C:\BusEngineRep\charge_class_classification\machine_learning\runClassificationPipeline.bat"