
import os, sys
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
main_categories=main_categories.split(",")
print 'Total',len(main_categories),"Wiki topics"
topic_list=[]

for grp in main_categories:
    try:
        pages = wikipedia.page(grp)
        links = pages.links
        print grp, 'has', len(links), 'links'
        topic_list.append((grp,str(len(links))))
    except Exception as e:
        continue

meta_file = codecs.open("../data/wiki_metadata.csv", "w", "utf-8")
output = map(lambda x: ",".join(x), topic_list)
meta_file.write("\n".join(output))
meta_file.close()