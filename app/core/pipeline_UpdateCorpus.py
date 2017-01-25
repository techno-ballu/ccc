import logging
import pandas as pd
import re
# import pypyodbc # unused

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))


def updateCorpus():
    # unable to move this import outside
    from app.core import extractChanges

    # handled all null checks
    logging.info(' : '+__file__+' : '+'In updateCorpus method.')
    dbConnect = extractChanges.DBconnect()
    data = dbConnect.getManualReviewData()

    # Return if no data received
    if(len(data['Original Description']))==0:
        logging.info(' : '+__file__+' : '+'No Changes detected from Manual review, No updates into the corpus')
        dbConnect.closeConnection()
        return

    logging.info(' : '+__file__+' : '+'Manual review data fetched from manual review detail table')

    # Remove comma from CorrectDesc entered by the user
    origDesc = data['Original Description']
    NormDesc = data['Normalized Description']
    data['Correct normalized description'] = data['Correct normalized description'].replace(',',' ')
    CorrectDesc = data['Correct normalized description'].str.replace(r' +',' ')
    Changes = data['Changes']

    SpellChanges = [None]*len(origDesc)
    AbbrChanges = [None]*len(origDesc)
    extractChanges = extractChanges.ExtractChanges()
    logging.info(' : '+__file__+' : '+'Extraction of Changes in normalized text started.')

    # try catch around getChanges and skip that description if exception arises :done
    for i in range(0,len(NormDesc)):
        # print i,NormDesc[i]
        if not isinstance(CorrectDesc[i],float):
            # NormDesc[i] = NormDesc[i].replace('_',' ')
            try:
                if Changes[i] is not None:
                    Changes[i] = Changes[i].strip()
                AbbrChanges[i],SpellChanges[i] = extractChanges.getChanges(origDesc[i].strip(),NormDesc[i].strip(),Changes[i],CorrectDesc[i].strip())
                # get surrounding word
                AbbrChanges[i],SpellChanges[i] = extractChanges.getSurroundingWord(AbbrChanges[i],SpellChanges[i],origDesc[i])
            except:
                logging.debug(' : '+__file__+' : '+'Unable to extract changes from the following description')
                logging.debug(' : '+__file__+' : '+'origDesc: '+str(origDesc[i].strip()) + ';NormDesc: '+ str(NormDesc[i].strip()) + ';Changes: '+str(Changes[i])+';CorrectedDesc: '+str(CorrectDesc[i].strip()))
                AbbrChanges[i],SpellChanges[i] = None,None
                logging.debug(' : '+__file__+' : '+'Continuing...')
                continue

    logging.info(' : '+__file__+' : '+'Changes in normalized text extracted.')
    data['SpellChanges']=SpellChanges
    data['AbbrChanges']=AbbrChanges

    if(SpellChanges.count(None)==len(SpellChanges) and AbbrChanges.count(None)==len(AbbrChanges)):
        logging.info(' : '+__file__+' : '+'No Changes detected to add into the corpus')
        dbConnect.closeConnection()
        return


    AbbrChangesList,SpellChangesList = extractChanges.getChangesList(AbbrChanges,SpellChanges,data['assignee'],data['date'])
    # Abbreviation_new = pd.DataFrame(AbbrChangesList,columns=['Abbreviation','Expansion','assignee','date'])
    # after adding surrounding word
    Abbreviation_new = pd.DataFrame(AbbrChangesList,columns=['Abbreviation','Expansion','SurroundingWord','assignee','date'])
    Abbreviation_new = Abbreviation_new[Abbreviation_new['Expansion'] != 'NoChange']
    # SpellChanges_new = pd.DataFrame(SpellChangesList,columns=['Word','Match','assignee','date'])
    # after adding surrounding word
    SpellChanges_new = pd.DataFrame(SpellChangesList,columns=['Word','Match','SurroundingWord','assignee','date'])
    SpellChanges_new = SpellChanges_new[SpellChanges_new['Match'] != 'NoChange']


    if(len(Abbreviation_new['Abbreviation'])>0):
        Abbreviation_all = dbConnect.getAbbreviationsFromTable()
        frames = [Abbreviation_all, Abbreviation_new]
        Abbreviation_all = pd.concat(frames)
        # Abbreviation_all = Abbreviation_all.drop_duplicates(cols = ['Abbreviation','Expansion'],take_last = True)
        # after adding surrounding word
        Abbreviation_all = Abbreviation_all.drop_duplicates(cols = ['Abbreviation','Expansion','SurroundingWord'],take_last = True)

        # Abbreviation_all.to_csv('../abbrNewDis.csv')
        dbConnect.updateAbbreviationTable(Abbreviation_all)
        logging.info(' : '+__file__+' : '+'Abbreviations updated in database table.')
        dbConnect.generateAbbrevReverseMapping()
        logging.info(' : '+__file__+' : '+'Abbreviation reverse mapping csv generated')
    else:
        logging.info(' : '+__file__+' : '+'No new Abbreviations found..')

    if(len(SpellChanges_new['Word'])>0):
        SpellChanges_all = dbConnect.getSpellCorectionsFromTable()
        frames = [SpellChanges_all, SpellChanges_new]
        SpellChanges_all = pd.concat(frames)
        # SpellChanges_all = SpellChanges_all.drop_duplicates(cols = ['Word','Match'],take_last = True)
        # after adding surrounding word
        SpellChanges_all = SpellChanges_all.drop_duplicates(cols = ['Word','Match','SurroundingWord'],take_last = True)

        # SpellChanges_all.drop_duplicates(cols = 'Word',take_last = True)
        # SpellChanges_all.to_csv('../spellNew.csv')
        dbConnect.updateSpellCorrectionTable(SpellChanges_all)
        logging.info(' : '+__file__+' : '+'Spelling corrections updated in database table.')
    else:
        logging.info(' : '+__file__+' : '+'No new Spell corrections found..')

    dbConnect.closeConnection()

if __name__ == "__main__":
    updateCorpus()