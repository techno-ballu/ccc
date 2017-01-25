import pandas as pd
import numpy as np
import pypyodbc
import re
from app.conf import dbconfig
from app.conf import appconfig
import logging

class DBconnect(object):
    def __init__(self):
        self.servername=dbconfig.SERVER_NAME
        self.databasename=dbconfig.DATABASENAME
        self.trainingTable=dbconfig.TRAINING_TABLE
        self.trainingMDTable=dbconfig.TRAINING_MD
        # self.claimLineInputTable=dbconfig.CLAIM_LN_INP_TAB
        self.origDescTable=dbconfig.DISTINCT_ORIG_DESC_TAB
        self.clsfTempTable=dbconfig.CLASSIFICATION_TEMP_TAB
        self.classfMDTable=dbconfig.CLASSF_MD_TAB
        self.classificationOutputTable=dbconfig.CLSF_OUTPUT_TAB
        self.unlearnDataTable=dbconfig.UNLEARN_DATA_TAB
        self.unlearnMDTable=dbconfig.UNLEARN_MD_TAB
        self.mrDetailTable=dbconfig.MR_DETAIL_TAB
        self.abbr_corpus = dbconfig.ABBR_CORPUS
        self.spell_corr_corpus = dbconfig.SPELL_CORR_CORPUS
        self.data_folder = appconfig.DATA_MODELS_LEARNING
        self.data_folder_classify = appconfig.MODELS_FOLDER_CLASSIFICATION
        self.SQL_write="Driver={SQL Server};Server="+self.servername+";Database="+self.databasename+";Trusted_Connection=True;"
        self.write_connection = pypyodbc.connect(self.SQL_write)
        self.cursor = self.write_connection.cursor()

    def closeConnection(self):
        self.cursor.close()
        self.write_connection.close()

    def getManualReviewData(self):
        SQL_query=("SELECT ORIG_DESC,NORM_DESC_ORIG,CHNGS,NORM_DESC_NEW,UpdatedBy,UpdatedDate FROM "+ self.mrDetailTable +" WHERE NORM_DESC_NEW <> NORM_DESC_ORIG and NORM_DESC_ORIG is not NULL and REV_STAT = 'Approved'")

        self.cursor.execute(SQL_query)
        ChangesData=pd.DataFrame(self.cursor.fetchall(),columns=['Original Description','Normalized Description','Changes','Correct normalized description','assignee','date'])
        self.cursor.commit()
        return ChangesData

    def getAbbreviationsFromTable(self):
        # SQL_query=("SELECT abbr,full_form,assignee,create_dt FROM " + self.abbr_corpus + " order by create_dt")
        # after adding surrounding words
        SQL_query=("SELECT abbr,full_form,surrounding_word,assignee,create_dt FROM " + self.abbr_corpus + " order by create_dt")

        self.cursor.execute(SQL_query)
        # Abbreviations=pd.DataFrame(self.cursor.fetchall(),columns=['Abbreviation','Expansion','assignee','date'])
        # after adding surrounding words
        Abbreviations=pd.DataFrame(self.cursor.fetchall(),columns=['Abbreviation','Expansion','SurroundingWord','assignee','date'])
        self.cursor.commit()
        return Abbreviations

    def getSpellCorectionsFromTable(self):
        # SQL_query=("SELECT incorr_spell,corr_spell,assignee,create_dt FROM " + self.spell_corr_corpus + " order by create_dt")
        # after adding surrounding words
        SQL_query=("SELECT incorr_spell,corr_spell,surrounding_word,assignee,create_dt FROM " + self.spell_corr_corpus + " order by create_dt")

        self.cursor.execute(SQL_query)
        # SpellChanges=pd.DataFrame(self.cursor.fetchall(),columns=['Word','Match','assignee','date'])
        # after adding surrounding words
        SpellChanges=pd.DataFrame(self.cursor.fetchall(),columns=['Word','Match','SurroundingWord','assignee','date'])
        self.cursor.commit()
        return SpellChanges

    def updateAbbreviationTable(self,Abbreviation_all):
        SQL_query=("TRUNCATE Table " + self.abbr_corpus)
        self.cursor.executemany(SQL_query)
        self.cursor.commit()

        # SQL_query=("INSERT INTO " + self.abbr_corpus + " "
        #              "(abbr,full_form,"
        #                 "assignee,create_dt)"
        #              "VALUES (?,?,?,?)")

        # after adding surrounding words
        SQL_query=("INSERT INTO " + self.abbr_corpus + " "
                     "(abbr,full_form,surrounding_word,"
                        "assignee,create_dt)"
                     "VALUES (?,?,?,?,?)")

        AbbrList=Abbreviation_all.values.tolist()
        self.cursor.executemany(SQL_query, AbbrList)
        self.cursor.commit()

    def generateAbbrevReverseMapping(self):
        SQL_query=("SELECT abbr,full_form,assignee,create_dt FROM " + self.abbr_corpus)

        self.cursor.execute(SQL_query)
        Abbreviations=pd.DataFrame(self.cursor.fetchall(),columns=['Abbreviation','Expansion','assignee','date'])
        self.cursor.commit()
        Abbreviations['Expansion'] = Abbreviations['Expansion'].str.lower()
        Abbreviations['Expansion'] = Abbreviations['Expansion'].str.strip()
        Abbreviations['Expansion'] = Abbreviations['Expansion'].str.replace(' ','_')
        Abbreviations['Abbreviation'] = Abbreviations['Abbreviation'].str.lower()
        Abbreviations[['Expansion','Abbreviation']].to_csv('../' + self.data_folder_classify + '/Abbreviations_ReverseMapping.csv',index=False)
        return Abbreviations[['Expansion','Abbreviation']]

    def updateSpellCorrectionTable(self,SpellChanges_all):
        SQL_query=("TRUNCATE Table " + self.spell_corr_corpus)
        self.cursor.executemany(SQL_query)
        self.cursor.commit()

        # SQL_query=("INSERT INTO " + self.spell_corr_corpus + " "
        #              "(incorr_spell, corr_spell,"
        #                 "assignee,create_dt)"
        #              "VALUES (?,?,?,?)")

        # after adding surrounding words
        SQL_query=("INSERT INTO " + self.spell_corr_corpus + " "
                     "(incorr_spell, corr_spell,surrounding_word,"
                        "assignee,create_dt)"
                     "VALUES (?,?,?,?,?)")

        SpellChangesList=SpellChanges_all.values.tolist()
        self.cursor.executemany(SQL_query, SpellChangesList)
        self.cursor.commit()


class ExtractChanges(object):
    def __init__(self):
        self.data_folder = appconfig.DATA_MODELS_LEARNING
        # commenting for testing with empty abbreviations list
        self.abbreviations_reverse_mapping = self.AbbrevReverseMap()
        self.abbreviations = list(self.abbreviations_reverse_mapping.index.values)

        # self.abbreviations_reverse_mapping = []
        # self.abbreviations = []
        self.NormChangesDict = {}

    def AbbrevReverseMap(self):
        dbconnect = DBconnect()
        df = dbconnect.generateAbbrevReverseMapping()
        dbconnect.closeConnection()
        return df

    # handled exceptions in this method at the higher level i.e in pipeline_UpdateCorpus
    def getChanges(self,origDesc, NormDesc, NormChanges, CorrectDesc):
        logging.info('In getChanges() method')
        origDesc = origDesc.lower()
        NormDesc = NormDesc.lower()
        CorrectDesc = CorrectDesc.lower()
        CorrectDesc = re.sub('[^A-Za-z0-9+%.>/&]',' ',CorrectDesc)
        CorrectDesc = re.sub('[ ]{2,}',' ',CorrectDesc)
        NormDesc = NormDesc.replace('/',' ')
        CorrectDesc = CorrectDesc.replace('/',' ')
        logging.debug('Before NormaChangesDict assign:'+origDesc+':'+NormDesc+':'+str(NormChanges)+':'+CorrectDesc)
        logging.debug('\n')
        self.NormChangesDict = self.getNormalizedChangesDF(NormChanges)
        logging.debug('After NormaChangesDict assign:'+str(type((self.NormChangesDict))))
        # NormDesc[i] = NormDesc[i].replace('_',' ')
        SpellChanges = None
        AbbrChanges = None
        origTokens = origDesc.split(' ')
        NormTokens = NormDesc.split(' ')
        CorrectedTokens = CorrectDesc.split(' ')
        arr = [0]*len(CorrectedTokens)
        k,found = -1,0
        logging.info('Identifying changes.....')
        for j in range(0, len(NormTokens)):
            logging.debug(j)
            found = 0
            logging.debug(NormTokens[j])
            nToken = NormTokens[j]
            if len(nToken)>1 and nToken in CorrectedTokens:                        # If it is correct
                arr[CorrectedTokens.index(nToken)]=1
                k = CorrectedTokens.index(nToken)
            elif len(nToken)>1:
                if nToken.isdigit():                                    # If it is a number
                    continue
                if '_' in nToken:                                                                   # and (nToken in self.abbreviations):
                    logging.debug('underscore in nToken..')
                    if nToken.replace("_"," ") in CorrectDesc or nToken.replace("_","") in CorrectDesc:
                        logging.debug('abbrev present in CorrectDesc')
                        arr[k+1:k+len(nToken.split("_"))+1] = [2]*(len(nToken.split("_")))
                        k += len(nToken.split("_"))
                        logging.debug('array updated after: abbrev present in CorrectDesc')
                        continue
                    else:
                        miniTokens = nToken.split("_")

                        for m in range(k+1,len(CorrectedTokens)-1):
                            logging.debug('checking for a new Abbrev')
                            if len(miniTokens[0]) > 0 and len(miniTokens[1]) > 0  and len(CorrectedTokens[m]) > 0 and len(CorrectedTokens[m+1]) > 0 :
                                
                                if ((miniTokens[0])[0]==(CorrectedTokens[m])[0]) and ((miniTokens[1])[0]==(CorrectedTokens[m+1])[0]):
                                    
                                    correctAbbrev = '_'.join(CorrectedTokens[m:m+len(miniTokens)])
                                    if self.checkCorrectMatch(j,NormDesc,k,k+m+len(miniTokens)-1,CorrectDesc):
                                        logging.debug('found new Abbr/Spell')
                                        AbbrChanges,SpellChanges = self.appendMissingTokens(arr,m,AbbrChanges,SpellChanges,origDesc,CorrectedTokens)
                                        AbbrChanges,SpellChanges = self.appendChanges(AbbrChanges,SpellChanges,nToken,correctAbbrev,origDesc)
                                        arr[m:m+len(nToken.split("_"))] = [1]*(len(nToken.split("_")))
                                        k += m+len(miniTokens)-1
                                        found = 1
                                        logging.debug('updated arrays after finding new Abbr/Spell')
                                    break
                if found == 1:
                    continue
                for l in range(k+1,len(CorrectedTokens)):
                    logging.debug('finding correct match')
                    if nToken[0:1] == CorrectedTokens[l][0:1]:
                        # print nToken,CorrectedTokens[l],j,NormDesc,k,l,CorrectDesc
                        AbbrChanges,SpellChanges = self.appendMissingTokens(arr,l,AbbrChanges,SpellChanges,origDesc,CorrectedTokens)
                        if self.checkCorrectMatch(j,NormDesc,k,l,CorrectDesc):
                            AbbrChanges,SpellChanges = self.appendChanges(AbbrChanges,SpellChanges,nToken,CorrectedTokens[l],origDesc)

                        found = 1
                        k=l
                        arr[l]=1
                        logging.debug('found correct match')
                        break
                if found == 0:
                    AbbrChanges,SpellChanges = self.appendMissingTokens(arr,k,AbbrChanges,SpellChanges,origDesc,CorrectedTokens)
                    AbbrChanges,SpellChanges = self.appendChanges(AbbrChanges,SpellChanges,nToken,"NoChange",origDesc)
                    logging.debug('Handled when correct match not found')

        AbbrChanges,SpellChanges = self.appendMissingTokens(arr,len(CorrectedTokens),AbbrChanges,SpellChanges,origDesc,CorrectedTokens)
        logging.info('Changes identified.')
        return self.rearrange(AbbrChanges,SpellChanges)



    def appendChanges(self,AbbrChanges,SpellChanges,nToken,change,origDesc):
        logging.info('In appendChanges() method.')
        originalToken = nToken
        if self.NormChangesDict.has_key(nToken):
            originalToken = self.NormChangesDict[nToken]

        if nToken in self.abbreviations:
            # abbr_exps = np.array_str(self.abbreviations_reverse_mapping.loc[nToken].values)


            abbr_exps = [str(abbr) for abbr in self.abbreviations_reverse_mapping.loc[nToken].values]
            # testing
            # abbr_exps = [str(abbr) for abbr in []]

            for abbr in abbr_exps:
                if abbr.lower() in origDesc:
                    if AbbrChanges is not None:
                        AbbrChanges = AbbrChanges + abbr + ':' + change+';'
                    else:
                        AbbrChanges = abbr+':'+change+';'
                    return AbbrChanges,SpellChanges

        if SpellChanges is not None:
            SpellChanges = SpellChanges + originalToken + ':' + change+';'
        else:
            SpellChanges = originalToken+':'+change+';'
        return AbbrChanges,SpellChanges

    def checkCorrectMatch(self,normIndex,NormDesc,k,correctIndex,CorrectDesc):
        logging.info('In checkCorrectMatch() method.')
        NormTokens = NormDesc.split(' ')
        CorrectedTokens = CorrectDesc.split(' ')
        if normIndex == len(NormTokens)-1:
            return True
        nextToken = NormTokens[normIndex+1]

        if nextToken in CorrectedTokens:
            if CorrectedTokens.index(nextToken) > correctIndex:
                return True

        if nextToken.isdigit():                                    # If it is a number
            return self.checkCorrectMatch(normIndex+1,NormDesc,k,correctIndex,CorrectDesc)
        # if '_' in nextToken:                                                                   # and (nToken in self.abbreviations):   # change this (not completed)
        #     if nextToken.replace("_"," ") in CorrectDesc or nextToken.replace("_","") in CorrectDesc:
        #         k += len(nextToken.split("_"))
        #         continue
        #     else:
        #         miniTokens = nToken.split("_")
        #         for m in range(k+1,len(CorrectedTokens)-1):
        #             if ((miniTokens[0])[0]==(CorrectedTokens[m])[0]) and ((miniTokens[1])[0]==(CorrectedTokens[m+1])[0]):
        #                 correctAbbrev = '_'.join(CorrectedTokens[m:m+len(miniTokens)])
        #                 AbbrChanges,Changes = self.appendChanges(AbbrChanges,Changes,nToken,correctAbbrev,origDesc)
        #
        #                 arr[m:m+len(nToken.split("_"))] = [1]*(len(nToken.split("_")))
        #                 k += m+len(miniTokens)-1
        #                 found = 1
        #                 break
        # if found == 1:
        #     continue
        for l in range(k+1,len(CorrectedTokens)):                 # find the match for next token and see its position
            if nextToken[0:1] == CorrectedTokens[l][0:1]:
                found = 1
                if l > correctIndex:
                    return True
                else:
                    return False

        return True

    def appendMissingTokens(self,arr,index,AbbrChanges,SpellChanges,origDesc,CorrectedTokens):
        logging.info('In appendMissingTokens method.')
        if index==0:
            return AbbrChanges,SpellChanges
        for i in range(0,index):
            if arr[i]==0:
                if arr[i-1]==1 and CorrectedTokens[i] not in origDesc.split(' '):
                    if SpellChanges is not None and CorrectedTokens[i] != '':
                        SpellChanges = SpellChanges[0:len(SpellChanges)-1] + '_' + CorrectedTokens[i] + ';'
                        SpellChanges = SpellChanges.replace("NoChange_",'')
                    arr[i]=1
                if arr[i-1]==2 and CorrectedTokens[i] not in origDesc.split(' '):
                    if AbbrChanges is not None and CorrectedTokens[i] != '':
                        AbbrChanges = AbbrChanges[0:len(AbbrChanges)-1] + '_' + CorrectedTokens[i] + ';'
                        AbbrChanges = AbbrChanges.replace("NoChange_",'')
                    arr[i]=2

        return AbbrChanges,SpellChanges

    def rearrange(self,AbbrChanges,SpellChanges):
        logging.info('In rearrange() method.')
        if SpellChanges == None or '_' not in SpellChanges:
            return AbbrChanges,SpellChanges
        changesTokens = SpellChanges.split(';')
        SpellChanges = None
        for change in changesTokens:
            if change is not '':
                if '_' in change:
                    if AbbrChanges == None:
                        AbbrChanges = change + ';'
                    else:
                        AbbrChanges = AbbrChanges + change + ';'
                else:
                    if SpellChanges == None:
                        SpellChanges = change + ';'
                    else:
                        SpellChanges = SpellChanges + change + ';'
        return AbbrChanges,SpellChanges

    def getNormalizedChangesDF(self,NormChanges):
        logging.info('In getNormalizedChangesDF() method.')
        changesDict = {}
        if NormChanges is None:
            return ''
        for nChange in NormChanges.split('||'):
            if nChange != None and len(nChange)>1 and (':' in nChange):
                changesDict[nChange.split(':')[1]] = nChange.split(':')[0]
        return changesDict

    def getChangesList(self,AbbrChanges,SpellChanges,Assignee,Date):
        logging.info('In getChangesList() method.')
        AbbrList = []
        SpellList = []
        temp = []
        for i in range(0,len(AbbrChanges)):
            if AbbrChanges[i] is not None:
                for abbr in AbbrChanges[i].split(';'):
                    if abbr != None and len(abbr)>0:
                        temp = abbr.split(':')
                        temp.extend([Assignee[i],Date[i]])
                        AbbrList.append(temp)
            if SpellChanges[i] is not None:
                for spellChange in SpellChanges[i].split(';'):
                    if spellChange != None and len(spellChange)>0:
                        temp = spellChange.split(':')
                        temp.extend([Assignee[i],Date[i]])
                        SpellList.append(temp)

        return AbbrList,SpellList

    def getSurroundingWord(self,AbbrChanges,SpellChanges,origDesc):
        origTokens = origDesc.split(' ')
        # origTokens = re.split('/ ',origDesc)
        origTokensLength = len(origTokens)
        origTokensWithoutSlash = re.split('[/ ]',origDesc)
        origTokensWithoutSlashLength = len(origTokensWithoutSlash)

        AbbrChangesNew=''
        SpellChangesNew=''

        if AbbrChanges != None and len(AbbrChanges)>0:
            for abbrChange in AbbrChanges.split(';'):
                if abbrChange != None and len(abbrChange)>0:
                    origAbbr = abbrChange.split(':')[0]

                    try:
                        if origAbbr in origTokens:
                            origAbbrIndex = origTokens.index(origAbbr)
                            if origAbbrIndex==origTokensLength-1 and origTokensLength != 1:
                                AbbrChangesNew = AbbrChangesNew + ';' + abbrChange + ':' + origTokens[origTokensLength-2]
                            else:
                                AbbrChangesNew = AbbrChangesNew + ';' + abbrChange + ':' + origTokens[origAbbrIndex+1]
                        else:
                            origAbbrIndex = origTokensWithoutSlash.index(origAbbr)
                            if origAbbrIndex==origTokensWithoutSlashLength-1 and origTokensWithoutSlashLength != 1:
                                AbbrChangesNew = AbbrChangesNew + ';' + abbrChange + ':' + origTokensWithoutSlash[origTokensWithoutSlashLength-2]
                            else:
                                AbbrChangesNew = AbbrChangesNew + ';' + abbrChange + ':' + origTokensWithoutSlash[origAbbrIndex+1]


                    except:
                        AbbrChangesNew = AbbrChangesNew + ';' + abbrChange + ':'
            AbbrChangesNew = AbbrChangesNew[1:]

        if SpellChanges != None and len(SpellChanges)>0:
            for spellChange in SpellChanges.split(';'):
                if spellChange != None and len(spellChange)>0:
                    origSpell = spellChange.split(':')[0]

                    try:
                        if origSpell in origTokens:
                            origSpellIndex = origTokens.index(origSpell)
                            if origSpellIndex==origTokensLength-1 and origTokensLength != 1:
                                SpellChangesNew = SpellChangesNew + ';' + spellChange + ':' + origTokens[origTokensLength-2]
                            else:
                                SpellChangesNew = SpellChangesNew + ';' + spellChange + ':' + origTokens[origSpellIndex+1]
                        else:
                            origSpellIndex = origTokensWithoutSlash.index(origSpell)
                            if origSpellIndex==origTokensWithoutSlashLength-1 and origTokensWithoutSlashLength != 1:
                                SpellChangesNew = SpellChangesNew + ';' + spellChange + ':' + origTokensWithoutSlash[origTokensWithoutSlashLength-2]
                            else:
                                SpellChangesNew = SpellChangesNew + ';' + spellChange + ':' + origTokensWithoutSlash[origSpellIndex+1]
                    except:
                        SpellChangesNew = SpellChangesNew + ';' + spellChange + ':'

            SpellChangesNew = SpellChangesNew[1:]

        return AbbrChangesNew,SpellChangesNew

