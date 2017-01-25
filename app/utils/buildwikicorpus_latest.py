import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import codecs
import wikipedia
import warnings
import ConfigParser

config = ConfigParser.ConfigParser()
config.read("../conf/wikiconfig.py")
warnings.simplefilter("ignore")

main_categories = config.get('buildwikicorpus properties', 'main_categories')
data_location="../data/wikidata_new"
main_categories=main_categories.split(",")
print 'Total',len(main_categories),"Wiki topics"
list_articles = []
topic_list=[]

for grp in main_categories:
    file_name = os.path.join(data_location,'.txt' )
    list_articles.append((grp, grp))
    try:
        pages = wikipedia.page(grp)
        content = pages.content
        links = pages.links
        print grp, 'has', len(links), 'links'
        topic_list.append((grp,len(links)))
        file_wiki = codecs.open(file_name, "w", "utf-8")
        file_wiki.write(content)
        file_wiki.close()
    except Exception as e:
        #print e
        #print grp + ' could not be retrieved!'
        continue
    for l in links:
        file_name = os.path.join(data_location, l+'.txt' )
        file_present = os.path.isfile(file_name)
        if file_present:
            #print l+' is already extracted'
            continue
        else:
            try:
                #print 'Extracting: ', grp, '>', l
                page = wikipedia.page(l)
                content = page.content
                file_wiki = codecs.open(file_name, "w", "utf-8")
                file_wiki.write(content)
                file_wiki.close()
                list_articles.append((grp,l))
            except wikipedia.exceptions.PageError as pe:
                #print pe
                #print l + ' could not be retrieved!'
                continue
            except wikipedia.exceptions.DisambiguationError as de:
                #print de
                #print l + ' could not be retrieved!'
                continue
            except Exception as e:
                #print e
                #print l + ' could not be retrieved!'
                continue

print
print 'Total ', len(list_articles), 'links collected!'
meta_file = codecs.open("../data/wiki_metadata.csv", "w", "utf-8")
meta_file.write(meta_file)
meta_file.close()